#!/usr/bin/env ruby
# frozen_string_literal: true

# TurboRAG Ruby Client Benchmark with Memory Tracking

require_relative "lib/turborag"
require "json"

BASE_URL = ENV.fetch("TURBORAG_URL", "http://127.0.0.1:8080")
QUERY_COUNT = ENV.fetch("TURBORAG_NUM_QUERIES", "50").to_i
TOP_K = ENV.fetch("TURBORAG_TOP_K", "10").to_i
INGEST_COUNT = ENV.fetch("TURBORAG_INGEST_RECORDS", "100").to_i

def make_vector(dim, seed)
  values = (0...dim).map do |i|
    Math.sin(seed + i * 0.07) + Math.cos(seed * 0.11 + i * 0.03)
  end
  norm = Math.sqrt(values.sum { |v| v * v })
  norm = 1.0 if norm.zero?
  values.map { |v| v / norm }
end

def format_bytes(bytes)
  if bytes < 1024
    "#{bytes} B"
  elsif bytes < 1024 * 1024
    "#{(bytes / 1024.0).round(2)} KB"
  else
    "#{(bytes / (1024.0 * 1024)).round(2)} MB"
  end
end

def get_memory_usage
  # Use /proc/self/status on Linux, or ps on macOS
  if File.exist?("/proc/self/status")
    status = File.read("/proc/self/status")
    rss_kb = status.match(/VmRSS:\s+(\d+)/)[1].to_i
    rss_kb * 1024
  else
    # macOS fallback using ps
    pid = Process.pid
    rss_kb = `ps -o rss= -p #{pid}`.strip.to_i
    rss_kb * 1024
  end
rescue
  0
end

# Force GC before starting
GC.start

mem_start = get_memory_usage
max_rss = mem_start

client = TurboRAG::Client.new(BASE_URL, timeout: 120)
index = client.index

dim = index["dim"]
queries = (1..QUERY_COUNT).map { |i| make_vector(dim, i.to_f) }

puts "\n=== TurboRAG Ruby Client Benchmark ===\n\n"
puts "Base URL: #{BASE_URL}"
puts "Index size: #{index["index_size"]}"
puts "Dimension: #{dim}"
puts "Bits: #{index["bits"]}"
puts "Queries: #{QUERY_COUNT}"
puts "topK: #{TOP_K}"
puts "\nMemory at start: #{format_bytes(mem_start)}\n\n"

# Single query benchmark with memory tracking
single_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
queries.each do |vector|
  client.query(vector: vector, top_k: TOP_K)
  current_rss = get_memory_usage
  max_rss = [max_rss, current_rss].max
end
single_elapsed = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - single_start) * 1000
mem_after_single = get_memory_usage
max_rss = [max_rss, mem_after_single].max

# Batch query benchmark
batch_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
batch_response = client.query_batch(
  queries: queries.map { |v| { vector: v } },
  top_k: TOP_K
)
batch_elapsed = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - batch_start) * 1000
mem_after_batch = get_memory_usage
max_rss = [max_rss, mem_after_batch].max

# Ingest benchmark
records = (0...INGEST_COUNT).map do |i|
  {
    chunk_id: "ruby-bench-#{Time.now.to_i}-#{i}",
    text: "Benchmark record #{i}",
    embedding: make_vector(dim, 10_000 + i),
    metadata: { benchmark: true, index: i }
  }
end

ingest_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
ingest_response = client.ingest(records: records)
ingest_elapsed = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - ingest_start) * 1000
mem_after_ingest = get_memory_usage
max_rss = [max_rss, mem_after_ingest].max

results = {
  baseUrl: BASE_URL,
  index: {
    size: index["index_size"],
    dim: dim,
    bits: index["bits"]
  },
  singleQuery: {
    queries: QUERY_COUNT,
    elapsedMs: single_elapsed.round(2),
    qps: (QUERY_COUNT / (single_elapsed / 1000)).round(2),
    avgLatencyMs: (single_elapsed / QUERY_COUNT).round(2)
  },
  batchQuery: {
    queries: QUERY_COUNT,
    batchCount: batch_response["batch_count"],
    elapsedMs: batch_elapsed.round(2),
    qps: (QUERY_COUNT / (batch_elapsed / 1000)).round(2),
    avgLatencyMs: (batch_elapsed / QUERY_COUNT).round(2)
  },
  ingest: {
    records: INGEST_COUNT,
    added: ingest_response["added"],
    indexSize: ingest_response["index_size"],
    elapsedMs: ingest_elapsed.round(2),
    recordsPerSecond: (INGEST_COUNT / (ingest_elapsed / 1000)).round(2)
  },
  memory: {
    startRssMB: (mem_start / (1024.0 * 1024)).round(2),
    maxRssMB: (max_rss / (1024.0 * 1024)).round(2),
    finalRssMB: (mem_after_ingest / (1024.0 * 1024)).round(2),
    rssGrowthMB: ((mem_after_ingest - mem_start) / (1024.0 * 1024)).round(2)
  }
}

puts JSON.pretty_generate(results)

puts "\n=== Memory Summary ==="
puts "Start RSS:  #{format_bytes(mem_start)}"
puts "Max RSS:    #{format_bytes(max_rss)}"
puts "Final RSS:  #{format_bytes(mem_after_ingest)}"
puts "Growth:     #{format_bytes(mem_after_ingest - mem_start)}"
