class TurboRAGError extends Error {
    constructor(message, statusCode = null, details = null) {
        super(message);
        this.name = 'TurboRAGError';
        this.statusCode = statusCode;
        this.details = details;
    }
}

class ValidationError extends TurboRAGError {
    constructor(message, details = null) {
        super(message, 400, details);
        this.name = 'ValidationError';
    }
}

class ConnectionError extends TurboRAGError {
    constructor(message, details = null) {
        super(message, null, details);
        this.name = 'ConnectionError';
    }
}

class TimeoutError extends TurboRAGError {
    constructor(message, details = null) {
        super(message, null, details);
        this.name = 'TimeoutError';
    }
}

module.exports = {
    TurboRAGError,
    ValidationError,
    ConnectionError,
    TimeoutError
};
