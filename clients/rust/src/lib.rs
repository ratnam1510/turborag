use reqwest::blocking::Client as HttpClient;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::time::Duration;

#[derive(Debug)]
pub enum TurboRagError {
    Validation(String),
    Api { status: u16, body: String },
    Transport(reqwest::Error),
    InvalidHeader(reqwest::header::InvalidHeaderValue),
}

impl fmt::Display for TurboRagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(message) => write!(f, "{message}"),
            Self::Api { status, body } => write!(f, "TurboRAG HTTP {status}: {body}"),
            Self::Transport(error) => write!(f, "{error}"),
            Self::InvalidHeader(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TurboRagError {}

impl From<reqwest::Error> for TurboRagError {
    fn from(value: reqwest::Error) -> Self {
        Self::Transport(value)
    }
}

impl From<reqwest::header::InvalidHeaderValue> for TurboRagError {
    fn from(value: reqwest::header::InvalidHeaderValue) -> Self {
        Self::InvalidHeader(value)
    }
}

#[derive(Clone, Debug)]
pub struct ClientOptions {
    pub timeout: Duration,
    pub default_top_k: usize,
    pub headers: HeaderMap,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            default_top_k: 5,
            headers: HeaderMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TurboRagClient {
    base_url: String,
    http: HttpClient,
    default_top_k: usize,
    headers: HeaderMap,
}

impl TurboRagClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self, TurboRagError> {
        Self::with_options(base_url, ClientOptions::default())
    }

    pub fn from_env() -> Result<Self, TurboRagError> {
        let base_url = env::var("TURBORAG_API_URL")
            .or_else(|_| env::var("TURBORAG_URL"))
            .unwrap_or_else(|_| "http://localhost:8080".to_string());

        let timeout = env::var("TURBORAG_TIMEOUT_MS")
            .ok()
            .or_else(|| env::var("TURBORAG_TIMEOUT").ok())
            .and_then(|raw| raw.parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or_else(|| Duration::from_secs(30));

        let default_top_k = env::var("TURBORAG_TOP_K")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(5);

        Self::with_options(
            base_url,
            ClientOptions {
                timeout,
                default_top_k,
                headers: HeaderMap::new(),
            },
        )
    }

    pub fn with_options(
        base_url: impl Into<String>,
        options: ClientOptions,
    ) -> Result<Self, TurboRagError> {
        let http = HttpClient::builder().timeout(options.timeout).build()?;
        Ok(Self {
            base_url: normalize_base_url(base_url.into()),
            http,
            default_top_k: options.default_top_k,
            headers: options.headers,
        })
    }

    pub fn with_header(
        mut self,
        name: impl AsRef<str>,
        value: impl AsRef<str>,
    ) -> Result<Self, TurboRagError> {
        let header_name = HeaderName::from_bytes(name.as_ref().as_bytes())
            .map_err(|error| TurboRagError::Validation(error.to_string()))?;
        let header_value = HeaderValue::from_str(value.as_ref())?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    pub fn health(&self) -> Result<HealthResponse, TurboRagError> {
        self.get("/health")
    }

    pub fn index(&self) -> Result<IndexInfo, TurboRagError> {
        self.get("/index")
    }

    pub fn metrics(&self) -> Result<MetricsResponse, TurboRagError> {
        self.get("/metrics")
    }

    pub fn query(&self, request: QueryRequest) -> Result<QueryResponse, TurboRagError> {
        validate_vector(&request.vector, "vector")?;
        let top_k = self.resolve_top_k(request.top_k)?;
        let payload = QueryPayload {
            query_vector: Some(request.vector),
            query_text: None,
            top_k,
            hydrate: request.hydrate,
            filters: request.filters,
        };
        self.post("/query", &payload)
    }

    pub fn query_ids(&self, request: QueryRequest) -> Result<QueryResponse, TurboRagError> {
        self.query(QueryRequest {
            hydrate: Some(false),
            ..request
        })
    }

    pub fn query_text(
        &self,
        request: TextQueryRequest,
    ) -> Result<QueryResponse, TurboRagError> {
        validate_text(&request.text, "text")?;
        let top_k = self.resolve_top_k(request.top_k)?;
        let payload = QueryPayload {
            query_vector: None,
            query_text: Some(request.text),
            top_k,
            hydrate: request.hydrate,
            filters: request.filters,
        };
        self.post("/query", &payload)
    }

    pub fn query_batch(
        &self,
        request: BatchQueryRequest,
    ) -> Result<BatchQueryResponse, TurboRagError> {
        if request.queries.is_empty() {
            return Err(TurboRagError::Validation(
                "queries must be a non-empty array".to_string(),
            ));
        }
        let top_k = self.resolve_top_k(request.top_k)?;
        let queries = request
            .queries
            .into_iter()
            .map(|query| {
                validate_vector(&query.vector, "queries[].vector")?;
                Ok(BatchQueryPayloadItem {
                    query_vector: query.vector,
                })
            })
            .collect::<Result<Vec<_>, TurboRagError>>()?;

        let payload = BatchQueryPayload {
            queries,
            top_k,
            hydrate: request.hydrate,
        };
        self.post("/query/batch", &payload)
    }

    pub fn query_batch_ids(
        &self,
        request: BatchQueryRequest,
    ) -> Result<BatchQueryResponse, TurboRagError> {
        self.query_batch(BatchQueryRequest {
            hydrate: Some(false),
            ..request
        })
    }

    pub fn ingest(&self, request: IngestRequest) -> Result<IngestResponse, TurboRagError> {
        if request.records.is_empty() {
            return Err(TurboRagError::Validation(
                "records must be a non-empty array".to_string(),
            ));
        }
        for (index, record) in request.records.iter().enumerate() {
            if record.chunk_id.trim().is_empty() {
                return Err(TurboRagError::Validation(format!(
                    "records[{index}].chunk_id must be a non-empty string"
                )));
            }
            validate_vector(&record.embedding, &format!("records[{index}].embedding"))?;
        }
        self.post("/ingest", &request)
    }

    pub fn ingest_text(
        &self,
        request: IngestTextRequest,
    ) -> Result<IngestTextResponse, TurboRagError> {
        validate_text(&request.text, "text")?;
        self.post("/ingest-text", &request)
    }

    fn resolve_top_k(&self, top_k: Option<usize>) -> Result<usize, TurboRagError> {
        let value = top_k.unwrap_or(self.default_top_k);
        if !(1..=1000).contains(&value) {
            return Err(TurboRagError::Validation(
                "top_k must be between 1 and 1000".to_string(),
            ));
        }
        Ok(value)
    }

    fn get<T>(&self, path: &str) -> Result<T, TurboRagError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let response = self
            .http
            .get(format!("{}{}", self.base_url, path))
            .headers(self.headers.clone())
            .send()?;
        decode_response(response)
    }

    fn post<B, T>(&self, path: &str, body: &B) -> Result<T, TurboRagError>
    where
        B: Serialize + ?Sized,
        T: for<'de> Deserialize<'de>,
    {
        let mut headers = self.headers.clone();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let response = self
            .http
            .post(format!("{}{}", self.base_url, path))
            .headers(headers)
            .json(body)
            .send()?;
        decode_response(response)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct QueryRequest {
    pub vector: Vec<f32>,
    pub top_k: Option<usize>,
    pub hydrate: Option<bool>,
    pub filters: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TextQueryRequest {
    pub text: String,
    pub top_k: Option<usize>,
    pub hydrate: Option<bool>,
    pub filters: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BatchQueryItem {
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BatchQueryRequest {
    pub queries: Vec<BatchQueryItem>,
    pub top_k: Option<usize>,
    pub hydrate: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestRecord {
    pub chunk_id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_doc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_num: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IngestRequest {
    pub records: Vec<IngestRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_overlap: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IngestTextRequest {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_doc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_config: Option<ChunkConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QueryResult {
    pub chunk_id: String,
    pub text: String,
    pub score: f64,
    pub source_doc: Option<String>,
    pub page_num: Option<i32>,
    pub graph_path: Option<Vec<String>>,
    pub explanation: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QueryResponse {
    pub count: usize,
    pub results: Vec<QueryResult>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BatchQueryResult {
    pub count: usize,
    pub results: Vec<QueryResult>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BatchQueryResponse {
    pub batch_count: usize,
    pub results: Vec<BatchQueryResult>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IngestResponse {
    pub added: usize,
    pub index_size: usize,
    pub records_snapshot: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IngestTextChunk {
    pub chunk_id: String,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IngestTextResponse {
    pub added: usize,
    pub chunks: Vec<IngestTextChunk>,
    pub index_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub index_path: String,
    pub index_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IndexInfo {
    pub index_path: String,
    pub dim: usize,
    pub bits: usize,
    pub shard_size: usize,
    pub normalize: bool,
    pub value_range: f64,
    pub index_size: usize,
    pub records_loaded: usize,
    pub records_snapshot: Option<String>,
    pub text_query_enabled: bool,
    pub hydration_source: Option<String>,
    pub allow_unhydrated: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MetricsResponse {
    pub uptime_seconds: f64,
    pub errors: usize,
    pub endpoints: HashMap<String, MetricsEndpoint>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MetricsEndpoint {
    pub count: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
    pub min_ms: Option<f64>,
    pub max_ms: Option<f64>,
}

#[derive(Serialize)]
struct QueryPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    query_vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    query_text: Option<String>,
    top_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    hydrate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filters: Option<Value>,
}

#[derive(Serialize)]
struct BatchQueryPayload {
    queries: Vec<BatchQueryPayloadItem>,
    top_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    hydrate: Option<bool>,
}

#[derive(Serialize)]
struct BatchQueryPayloadItem {
    query_vector: Vec<f32>,
}

fn validate_vector(vector: &[f32], label: &str) -> Result<(), TurboRagError> {
    if vector.is_empty() {
        return Err(TurboRagError::Validation(format!(
            "{label} must be a non-empty array"
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(TurboRagError::Validation(format!(
            "{label} must contain only finite numeric values"
        )));
    }
    Ok(())
}

fn validate_text(text: &str, label: &str) -> Result<(), TurboRagError> {
    if text.trim().is_empty() {
        return Err(TurboRagError::Validation(format!(
            "{label} must be a non-empty string"
        )));
    }
    Ok(())
}

fn normalize_base_url(base_url: String) -> String {
    base_url.trim_end_matches('/').to_string()
}

fn decode_response<T>(response: reqwest::blocking::Response) -> Result<T, TurboRagError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = response.status();
    if status.is_success() {
        return response.json().map_err(TurboRagError::Transport);
    }

    let body = response.text().unwrap_or_default();
    Err(TurboRagError::Api {
        status: status.as_u16(),
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn query_posts_expected_payload() {
        let (base_url, receiver) = spawn_test_server(
            1,
            r#"{"count":1,"results":[{"chunk_id":"a","text":"apple","score":0.9,"source_doc":null,"page_num":null,"graph_path":null,"explanation":null}]}"#,
            StatusCode::OK,
        );

        let client = TurboRagClient::new(base_url).unwrap();
        let response = client
            .query(QueryRequest {
                vector: vec![0.1, 0.2, 0.3],
                top_k: Some(3),
                hydrate: Some(false),
                filters: Some(json!({"topic": "finance"})),
            })
            .unwrap();

        assert_eq!(response.count, 1);
        let request = receiver.recv().unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/query");
        let body: Value = serde_json::from_str(&request.body).unwrap();
        assert_eq!(body["query_vector"], json!([0.1, 0.2, 0.3]));
        assert_eq!(body["top_k"], 3);
        assert_eq!(body["hydrate"], false);
        assert_eq!(body["filters"], json!({"topic": "finance"}));
    }

    #[test]
    fn query_batch_rejects_empty_queries() {
        let client = TurboRagClient::new("http://localhost:8080").unwrap();
        let error = client
            .query_batch(BatchQueryRequest {
                queries: Vec::new(),
                top_k: Some(5),
                hydrate: None,
            })
            .unwrap_err();

        assert!(matches!(error, TurboRagError::Validation(_)));
    }

    #[test]
    fn ingest_posts_records() {
        let (base_url, receiver) = spawn_test_server(
            1,
            r#"{"added":1,"index_size":4,"records_snapshot":"/tmp/records.jsonl"}"#,
            StatusCode::OK,
        );

        let client = TurboRagClient::new(base_url).unwrap();
        let response = client
            .ingest(IngestRequest {
                records: vec![IngestRecord {
                    chunk_id: "doc-1".to_string(),
                    text: "finance".to_string(),
                    embedding: vec![0.3, 0.2, 0.1],
                    source_doc: Some("q1.pdf".to_string()),
                    page_num: None,
                    section: None,
                    metadata: Some(json!({"topic": "finance"})),
                }],
            })
            .unwrap();

        assert_eq!(response.added, 1);
        let request = receiver.recv().unwrap();
        assert_eq!(request.path, "/ingest");
        let body: Value = serde_json::from_str(&request.body).unwrap();
        assert_eq!(body["records"][0]["chunk_id"], "doc-1");
        assert_eq!(body["records"][0]["metadata"], json!({"topic": "finance"}));
    }

    #[test]
    fn api_errors_preserve_status_and_body() {
        let (base_url, _receiver) =
            spawn_test_server(1, r#"{"detail":"bad request"}"#, StatusCode::BAD_REQUEST);
        let client = TurboRagClient::new(base_url).unwrap();
        let error = client.health().unwrap_err();

        match error {
            TurboRagError::Api { status, body } => {
                assert_eq!(status, 400);
                assert!(body.contains("bad request"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[derive(Debug)]
    struct RecordedRequest {
        method: String,
        path: String,
        body: String,
    }

    fn spawn_test_server(
        expected_requests: usize,
        body: &'static str,
        status: StatusCode,
    ) -> (String, mpsc::Receiver<RecordedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();

        thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0_u8; 8192];
                let bytes_read = stream.read(&mut buffer).unwrap();
                let raw = String::from_utf8_lossy(&buffer[..bytes_read]).to_string();
                let mut parts = raw.split("\r\n\r\n");
                let head = parts.next().unwrap_or_default();
                let body_text = parts.next().unwrap_or_default().to_string();
                let mut lines = head.lines();
                let request_line = lines.next().unwrap_or_default();
                let mut request_parts = request_line.split_whitespace();
                let method = request_parts.next().unwrap_or_default().to_string();
                let path = request_parts.next().unwrap_or_default().to_string();
                sender
                    .send(RecordedRequest {
                        method,
                        path,
                        body: body_text,
                    })
                    .unwrap();

                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("OK"),
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        (format!("http://{address}"), receiver)
    }
}
