# frozen_string_literal: true

# TurboRAG Ruby client — typed wrapper over the TurboRAG HTTP API.
# Zero dependencies beyond stdlib (net/http, json).
#
# Usage:
#   client = TurboRAG::Client.new("http://localhost:8080")
#   results = client.query(vector: [0.1, 0.2, 0.3], top_k: 5)

require "net/http"
require "json"
require "uri"

module TurboRAG
  class Error < StandardError
    attr_reader :status, :body

    def initialize(status, body)
      @status = status
      @body = body
      super("TurboRAG HTTP #{status}: #{body}")
    end
  end

  class Client
    # @param base_url [String] TurboRAG server URL (e.g. "http://localhost:8080")
    # @param timeout [Integer] request timeout in seconds (default: 30)
    def initialize(base_url, timeout: 30)
      @base_uri = URI.parse(base_url.chomp("/"))
      @timeout = timeout
    end

    # Search by embedding vector.
    # @param vector [Array<Float>]
    # @param top_k [Integer]
    # @return [Hash] { "count" => N, "results" => [...] }
    def query(vector:, top_k: 5)
      post("/query", { query_vector: vector, top_k: top_k })
    end

    # Search by text (requires --model on the server).
    # @param text [String]
    # @param top_k [Integer]
    # @return [Hash]
    def query_text(text:, top_k: 5)
      post("/query", { query_text: text, top_k: top_k })
    end

    # Batch vector search.
    # @param queries [Array<Hash>] each with :vector key
    # @param top_k [Integer]
    # @return [Hash]
    def query_batch(queries:, top_k: 5)
      payload = {
        queries: queries.map { |q| { query_vector: q[:vector] } },
        top_k: top_k,
      }
      post("/query/batch", payload)
    end

    # Add records with precomputed embeddings.
    # @param records [Array<Hash>] each with :chunk_id, :text, :embedding
    # @return [Hash]
    def ingest(records:)
      post("/ingest", { records: records })
    end

    # Ingest raw text with auto-chunking (requires --model).
    # @param text [String]
    # @param source_doc [String, nil]
    # @return [Hash]
    def ingest_text(text:, source_doc: nil)
      post("/ingest-text", { text: text, source_doc: source_doc })
    end

    # Health check.
    # @return [Hash]
    def health
      get("/health")
    end

    # Index configuration and stats.
    # @return [Hash]
    def index
      get("/index")
    end

    # Latency and error metrics.
    # @return [Hash]
    def metrics
      get("/metrics")
    end

    private

    def get(path)
      uri = URI.join(@base_uri.to_s, path)
      req = Net::HTTP::Get.new(uri)
      execute(uri, req)
    end

    def post(path, body)
      uri = URI.join(@base_uri.to_s, path)
      req = Net::HTTP::Post.new(uri)
      req["Content-Type"] = "application/json"
      req.body = JSON.generate(body)
      execute(uri, req)
    end

    def execute(uri, req)
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = (uri.scheme == "https")
      http.open_timeout = @timeout
      http.read_timeout = @timeout

      res = http.request(req)
      raise Error.new(res.code.to_i, res.body) unless res.is_a?(Net::HTTPSuccess)

      JSON.parse(res.body)
    end
  end
end
