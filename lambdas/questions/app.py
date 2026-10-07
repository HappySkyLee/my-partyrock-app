import json
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

API_KEY = "23h1h3v1hbvhEgwqhij31sdasSsghvswp"

# Bedrock-supported MIME types
SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_DOC_MIMES   = {"application/pdf"}

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type, x-api-key",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def generate_questions():

    # ── 1. Auth ───────────────────────────────────────────────────────────
    if request.headers.get("x-api-key", "") != API_KEY:
        return Response("Unauthorized", status=401, headers=CORS_HEADERS)

    # ── 2. Parse body ─────────────────────────────────────────────────────
    body = request.get_json(force=True, silent=True)
    if not body:
        return Response("Invalid JSON body", status=400, headers=CORS_HEADERS)

    subject       = body.get("subject", "General")
    focus_topics  = body.get("focus_topics", "")
    difficulty    = body.get("difficulty", "Medium")
    num_questions = int(body.get("num_questions", 10))
    question_type = body.get("question_type", "MCQ")
    file_data     = body.get("file_data", "")   # raw base64, no data-URL prefix
    file_mime     = body.get("file_mime", "")

    # ── 3. Build prompt text ──────────────────────────────────────────────
    prompt_text = (
        f"You are an expert educator and exam designer. "
        f"Generate exactly {num_questions} exam questions of the type: {question_type}, "
        f"at a {difficulty} difficulty level for the subject: {subject}.\n\n"
    )
    if focus_topics:
        prompt_text += f"Prioritize these focus topics: {focus_topics}\n\n"

    prompt_text += (
        "**IMPORTANT FORMAT:**\n\n"
        f"Display questions ONE BY ONE, numbered sequentially:\n"
        f"  Question 1 of {num_questions}: <question text>\n"
        f"  ---\n"
        f"  Question 2 of {num_questions}: <question text>\n"
        f"  ---\n"
        f"  ... continue for all {num_questions} questions ...\n\n"
        "After ALL questions, add a clearly labelled **Answer Key** section:\n"
        "- MCQ: correct option letter + brief explanation\n"
        "- Short Answer: model answer (2-4 sentences)\n"
        "- Essay: key points that should be covered\n"
    )

    # ── 4. Build Bedrock content blocks ───────────────────────────────────
    content   = []
    file_used = False

    if file_data and file_mime:
        mime = file_mime.split(";")[0].strip().lower()

        if mime in SUPPORTED_IMAGE_MIMES:
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": mime,
                    "data":       file_data,
                },
            })
            content.append({
                "type": "text",
                "text": prompt_text + "\n\nLecture Content: [See attached image above]",
            })
            file_used = True

        elif mime in SUPPORTED_DOC_MIMES:
            # PDF document block — Bedrock reads the full text content
            content.append({
                "type": "document",
                "source": {
                    "type":       "base64",
                    "media_type": "application/pdf",
                    "data":       file_data,
                },
            })
            content.append({
                "type": "text",
                "text": prompt_text + "\n\nLecture Content: [See attached PDF document above]",
            })
            file_used = True
        # else: DOCX/PPTX/unknown → fall through to text-only below

    if not file_used:
        content.append({
            "type": "text",
            "text": (
                prompt_text
                + "\n\nNote: No supported lecture file was provided (or file type not supported). "
                  "Generate questions based on the subject and focus topics alone."
            ),
        })

    # ── 5. Validate with a test call before streaming ─────────────────────
    # Build the full request payload
    bedrock_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": content}],
    }

    # ── 6. Stream Bedrock response ────────────────────────────────────────
    def generate(payload):
        try:
            response = BEDROCK_CLIENT.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
            for event in response["body"]:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk["bytes"].decode("utf-8"))
                    chunk_type = chunk_data.get("type", "")

                    if chunk_type == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")

                    elif chunk_type == "message_delta":
                        # stop_reason arrives here — nothing to yield
                        pass

                    elif chunk_type == "error":
                        error_msg = chunk_data.get("error", {}).get("message", "Unknown Bedrock error")
                        yield f"\n\n[BEDROCK ERROR] {error_msg}"

        except BEDROCK_CLIENT.exceptions.ValidationException as e:
            yield f"\n\n[VALIDATION ERROR] {str(e)}"
        except BEDROCK_CLIENT.exceptions.AccessDeniedException as e:
            yield f"\n\n[ACCESS DENIED] Check Bedrock model access is enabled in ap-southeast-1. Detail: {str(e)}"
        except Exception as e:
            yield f"\n\n[ERROR] {type(e).__name__}: {str(e)}"

    return Response(
        stream_with_context(generate(bedrock_payload)),
        status=200,
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
