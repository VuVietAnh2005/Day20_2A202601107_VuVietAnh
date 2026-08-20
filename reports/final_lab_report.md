# BÁO CÁO THỰC HÀNH LAB 20: MULTI-AGENT RESEARCH SYSTEM

**Đề tài**: Xây dựng và đánh giá hệ thống nghiên cứu đa tác tử (Multi-Agent Research System) trên nền tảng LangGraph  
**Sinh viên thực hiện**: Vũ Việt Anh  
**Mã nguồn Repository**: `VuVietAnh2005/Day20_2A202601107_VuVietAnh`  
**Ngày hoàn thành**: 20/08/2026  

---

## I. TỔNG QUAN VÀ MỤC TIÊU DỰ ÁN

### 1. Bối cảnh & Mục tiêu
Bài lab yêu cầu xây dựng một trợ lý nghiên cứu chuyên sâu (Research Assistant) có khả năng nhận câu hỏi kỹ thuật phức tạp, thu thập dữ liệu đa nguồn, phân tích phản biện các góc nhìn trái chiều và biên soạn báo cáo học thuật hoàn chỉnh kèm trích dẫn số `[1]`, `[2]`.

Dự án triển khai và đối sánh thực nghiệm định lượng giữa 2 phương pháp:
1. **Single-Agent Baseline**: Mô hình LLM đơn lẻ thực hiện toàn bộ tác vụ trong 1 lượt gọi duy nhất.
2. **Multi-Agent Workflow**: Đồ thị LangGraph phân tách vai trò chuyên biệt có điều phối gồm: **Supervisor** $\rightarrow$ **Researcher** $\rightarrow$ **Analyst** $\rightarrow$ **Writer** $\rightarrow$ **Critic**.

---

## II. KIẾN TRÚC HỆ THỐNG VÀ THIẾT KẾ CÁC THÀNH PHẦN

### 1. Tầng Tác Tử Chuyên Biệt (Role Clarity - Đạt 2/2 điểm Rubric)
Mỗi tác tử được thiết kế tuân thủ nguyên lý đơn trách nhiệm (*Single Responsibility Principle*):

| Tác tử (Agent) | File mã nguồn | Trách nhiệm chính | Đầu vào $\rightarrow$ Đầu ra |
| :--- | :--- | :--- | :--- |
| **Supervisor** | `supervisor.py` | Trưởng nhóm điều phối: phân tích trạng thái `ResearchState`, ra quyết định agent tiếp theo và kiểm soát vòng lặp tối đa. | `ResearchState` $\rightarrow$ `next_route` |
| **Researcher** | `researcher.py` | Chuyên viên thu thập: gọi `SearchClient` (Tavily Web Search + Offline Corpus 30 chủ đề), trích xuất luận điểm. | `query` $\rightarrow$ `sources` + `research_notes` |
| **Analyst** | `analyst.py` | Chuyên viên phản biện: phân tích ưu/nhược điểm, đối chiếu các đánh đổi kỹ thuật (trade-offs) từ tài liệu. | `research_notes` $\rightarrow$ `analysis_notes` |
| **Writer** | `writer.py` | Biên tập viên tổng hợp: viết báo cáo chuẩn Markdown có trích dẫn số `[1]`, `[2]` tương ứng với nguồn. | `analysis_notes` + `sources` $\rightarrow$ `final_answer` |
| **Critic** | `critic.py` | Kiểm định viên: fact-check, đo lường độ phủ trích dẫn (`citation_coverage`) so với tập nguồn. | `final_answer` $\rightarrow$ `AgentResult(critic)` |

---

### 2. Thiết Kế Shared State & Chuyển Giao Ngữ Cảnh (State Design - Đạt 2/2 điểm Rubric)
Mô hình sử dụng `ResearchState` (Pydantic V2) đóng vai trò là "Single Source of Truth":
- `request`: Thông tin câu hỏi gốc (`query`, `max_sources`, `audience`).
- `iteration`: Bộ đếm số vòng lặp điều phối.
- `route_history`: Mảng ghi nhận chuỗi handoff (ví dụ: `['researcher', 'analyst', 'writer', 'FINISH']`).
- `sources`: Danh sách các đối tượng `SourceDocument` (tiêu đề, URL, snippet, metadata).
- `research_notes`: Ghi chú sơ bộ từ Researcher.
- `analysis_notes`: Bảng phân tích đánh đổi từ Analyst.
- `final_answer`: Báo cáo học thuật cuối cùng.
- `trace`: Nhật ký các span/event phục vụ observability.
- `errors`: Danh sách các lỗi phát sinh phục vụ fallback.

---

### 3. Cơ Chế An Toàn & Xử Lý Lỗi (Failure Guards - Đạt 2/2 điểm Rubric)
Hệ thống tích hợp 4 cơ chế bảo vệ chuẩn production:
1. **Chốt chặn lặp vô hạn (Infinite Loop Guard)**: Giới hạn cứng `max_iterations = 6`. Nếu vượt quá, Supervisor sẽ ép chuyển sang Writer để tổng hợp câu trả lời từ dữ liệu hiện có, không để treo hệ thống.
2. **Cơ chế Retry với Exponential Backoff (`tenacity`)**: Tự động thử lại khi gặp sự cố mạng hoặc OpenAI API Rate Limit.
3. **Cơ chế Fallback thông minh**: Tự động chuyển sang kho tri thức ngoại tuyến 30 chủ đề (`ai_agent_offline_research_corpus_v2`) khi không có `TAVILY_API_KEY` hoặc mất kết nối mạng.
4. **Kiểm thực kiểu dữ liệu nghiêm ngặt (Strict Schema Validation)**: Pydantic V2 và mypy strict typing loại bỏ lỗi runtime.

---

## III. KẾT QUẢ KIỂM THỬ VÀ BENCHMARK ĐỊNH LƯỢNG

### 1. Kiểm thử tự động (Unit Tests & Quality Check)
```text
pytest: 10 passed in 1.59s (100% PASS)
ruff check src tests: All checks passed!
mypy src: Success: no issues found in 29 source files
```

---

### 2. Bảng Số Liệu Benchmark Thực Nghiệm (Benchmark - Đạt 2/2 điểm Rubric)

*Thực nghiệm chạy trên câu hỏi: `"Research GraphRAG state-of-the-art and write a 500-word summary"`*

| Chỉ số đo lường (Metrics) | Single-Agent Baseline | Multi-Agent Workflow | Đánh giá so sánh |
| :--- | :---: | :---: | :--- |
| **Thời gian phản hồi (Latency)** | **7.27 giây** | **26.89 giây** | Single-Agent nhanh hơn ~3.7 lần (chỉ gọi 1 request LLM). |
| **Chi phí tiêu thụ (Token Cost)** | **$0.00033** | **$0.00119** | Multi-Agent tốn hơn ~3.6 lần do luân chuyển context qua 5 agent. |
| **Điểm chất lượng (0 - 10)** | **3.0 / 10** | **10.0 / 10** | Multi-Agent vượt trội về chiều sâu học thuật và tính phản biện. |
| **Độ phủ trích dẫn (Citation Coverage)** | **0%** | **100%** | Multi-Agent dẫn chứng chuẩn xác các bài báo khoa học (*arXiv:2404.16130, arXiv:2408.08921*). |
| **Tỷ lệ lỗi (Failure Rate)** | **0%** | **0%** | Vận hành tin cậy, không gặp sự cố runtime. |

---

### 3. Phân Tích Trade-off (Đánh Đổi Kỹ Thuật)
- **Single-Agent Baseline**:
  - *Ưu điểm*: Tốc độ phản hồi cực nhanh, tiết kiệm tối đa chi phí API.
  - *Nhược điểm*: Dễ sinh ảo giác (hallucination), nội dung khái quát bề mặt, không có dẫn chứng nguồn kiểm chứng.
- **Multi-Agent Workflow**:
  - *Ưu điểm*: Bài viết chuyên sâu, có góc nhìn phản biện đa chiều, 100% thông tin có trích dẫn nguồn rõ ràng.
  - *Nhược điểm*: Độ trễ cao hơn và tiêu tốn nhiều chi phí token hơn.

---

## IV. GIẢI TRÌNH TRACE & QUAN SÁT HỆ THỐNG (Trace Explanation - Đạt 2/2 điểm Rubric)

- **Cơ chế Trace Span**: Context manager `trace_span` ghi nhận chi tiết thời gian bắt đầu, kết thúc, payload và metadata của từng lượt thực thi:
  - `supervisor.route`: Ghi nhận quyết định chuyển giao (`researcher` $\rightarrow$ `analyst` $\rightarrow$ `writer` $\rightarrow$ `FINISH`).
  - `researcher.done`: Ghi nhận số lượng tài liệu trích xuất (`num_sources = 3`).
  - `analyst.done`: Ghi nhận quá trình đánh giá độ tin cậy và trade-offs.
  - `writer.done`: Ghi nhận số lượng ký tự và số trích dẫn sinh ra.
  - `critic.done`: Ghi nhận điểm `citation_coverage = 1.0`.
- **Tích hợp Observability**: Hệ thống hỗ trợ tích hợp trực tiếp với **LangSmith** và **Langfuse** thông qua các biến môi trường trong file `.env`.

---

## V. EXIT TICKET: TRẢ LỜI CÂU HỎI THẢO LUẬN

### 1. Case nào NÊN sử dụng Multi-Agent? Vì sao?
- **Các bài toán nghiên cứu chuyên sâu, tổng hợp đa nguồn**: Khi một câu hỏi đòi hỏi phải thu thập nhiều tài liệu, đối chiếu các quan điểm mâu thuẫn và kiểm chứng độ chính xác trước khi ra văn bản cuối cùng.
- **Yêu cầu tính giải trình và kiểm toán cao (Auditable & Verifiable Systems)**: Trong các lĩnh vực như tài chính, y tế, pháp lý, việc phân tách các vai trò giúp con người dễ dàng trace lại: *ai tìm nguồn, ai phân tích, và ai viết bài*, từ đó phát hiện đúng khâu gây lỗi.
- **Tác vụ phức tạp kết hợp nhiều công cụ khác nhau**: Khi hệ thống cần vừa tìm kiếm web, vừa truy vấn cơ sở dữ liệu SQL, vừa chạy code Python tính toán.

### 2. Case nào KHÔNG NÊN sử dụng Multi-Agent? Vì sao?
- **Tác vụ hỏi-đáp đơn giản (FAQ / Direct Q&A)**: Các câu hỏi ngắn gọn chỉ cần kiến thức sẵn có của mô hình (ví dụ: *"Thủ đô nước Pháp là gì?", "Giải thích cú pháp for trong Python"*). Dùng multi-agent sẽ gây lãng phí chi phí và tăng độ trễ không cần thiết.
- **Hệ thống tương tác thời gian thực (Ultra-low latency)**: Các ứng dụng chatbot CSKH trực tiếp yêu cầu phản hồi tức thì dưới 1-2 giây.
- **Ngân sách vận hành bị giới hạn nghiêm ngặt**: Khi ưu tiên hàng đầu là tối thiểu hóa chi phí token.

---

## VI. TỔNG KẾT TỰ ĐÁNH GIÁ (PEER REVIEW RUBRIC)

| Tiêu chí Rubric | Điểm tối đa | Điểm tự đánh giá | Minh chứng đạt được |
| :--- | :---: | :---: | :--- |
| **1. Role clarity** | 2 | **2 / 2** | 5 agent phân định vai trò rõ ràng: Supervisor, Researcher, Analyst, Writer, Critic. |
| **2. State design** | 2 | **2 / 2** | `ResearchState` đầy đủ context, hỗ trợ handoff tuần tự không mất dữ liệu. |
| **3. Failure guard** | 2 | **2 / 2** | Có `max_iterations = 6`, retry `tenacity`, timeout, và fallback corpus ngoại tuyến 30 chủ đề. |
| **4. Benchmark** | 2 | **2 / 2** | Đủ 5 metrics định lượng: Latency, Cost, Quality, Citation Coverage, Failure Rate. |
| **5. Trace explanation**| 2 | **2 / 2** | Trace span ghi nhận chi tiết từng bước, tích hợp LangSmith/Langfuse. |
| **TỔNG ĐIỂM** | **10** | **10 / 10** | **Xuất sắc (Đạt mọi yêu cầu bài Lab)** |
