"""
Parsing utilities for LLM outputs
"""

import re


def parse_llm_label(text: str) -> str:
    """Robustly extracts upregulated/downregulated from JSON output."""
    import json
    
    # Priority 1: Try to parse as JSON (expected format)
    # IMPORTANT: Only parse JSON from <|ASSISTANT|> tag onwards to avoid parsing prompt examples
    try:
        # Extract only the assistant's response (after <|ASSISTANT|> tag)
        assistant_start = text.find('<|ASSISTANT|>')
        if assistant_start != -1:
            assistant_text = text[assistant_start + len('<|ASSISTANT|>'):].strip()
        else:
            # If no ASSISTANT tag, use the whole text but try to find the last JSON
            assistant_text = text.strip()
        
        # Remove markdown code blocks if present
        cleaned_text = assistant_text
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Find ALL JSON objects in the text
        json_objects = []
        i = 0
        while i < len(cleaned_text):
            if cleaned_text[i] == '{':
                # Found start of JSON object, find matching closing brace
                brace_count = 0
                start_idx = i
                end_idx = i
                for j in range(i, len(cleaned_text)):
                    if cleaned_text[j] == '{':
                        brace_count += 1
                    elif cleaned_text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = j
                            break
                
                if end_idx > start_idx:
                    json_str = cleaned_text[start_idx:end_idx + 1]
                    try:
                        parsed = json.loads(json_str)
                        if "answer" in parsed:
                            json_objects.append((start_idx, parsed))
                    except json.JSONDecodeError:
                        pass
                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        # Use the LAST JSON object (most likely to be the actual answer, not prompt example)
        if json_objects:
            # Sort by position (last one is the actual answer)
            json_objects.sort(key=lambda x: x[0])
            _, parsed = json_objects[-1]  # Get the last JSON object
            
            answer = parsed["answer"].strip().lower()
            
            # Handle case where answer might be wrapped in <answer> tags
            # e.g., answer = "<answer>upregulated</answer>" or answer = "upregulated</answer>"
            # Remove any XML-like tags from answer
            answer_clean = re.sub(r'<[^>]+>', '', answer).strip()
            
            # Also try to extract from <answer> tags if present
            answer_match = re.search(r'<answer>(upregulated|downregulated)</answer>', answer, re.IGNORECASE)
            if answer_match:
                answer_clean = answer_match.group(1).lower()
                print(f"🔍 [PARSER] JSON parse successful (extracted from <answer> tag in JSON): '{answer_clean}'")
                return answer_clean
            
            # Use cleaned answer (with tags removed)
            if answer_clean in ["upregulated", "downregulated"]:
                if answer_clean != answer:
                    print(f"🔍 [PARSER] JSON parse successful (cleaned tags from answer): '{answer_clean}' (was: '{answer}')")
                else:
                    print(f"🔍 [PARSER] JSON parse successful (from last JSON object): '{answer_clean}'")
                return answer_clean
            else:
                print(f"🔍 [PARSER] JSON parse found invalid answer: '{answer}' (cleaned: '{answer_clean}'), returning 'uncertain'")
                return "uncertain"
        
        # Fallback: Try to find any JSON object (if no ASSISTANT tag found)
        start_idx = cleaned_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(cleaned_text)):
                if cleaned_text[i] == '{':
                    brace_count += 1
                elif cleaned_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if end_idx > start_idx:
                json_str = cleaned_text[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                
                if "answer" in parsed:
                    answer = parsed["answer"].strip().lower()
                    
                    # Handle case where answer might be wrapped in <answer> tags
                    answer_match = re.search(r'<answer>(upregulated|downregulated)</answer>', answer, re.IGNORECASE)
                    if answer_match:
                        answer = answer_match.group(1).lower()
                        print(f"🔍 [PARSER] JSON parse successful (fallback, extracted from <answer> tag): '{answer}'")
                        return answer
                    
                    if answer in ["upregulated", "downregulated"]:
                        print(f"🔍 [PARSER] JSON parse successful (fallback): '{answer}'")
                        return answer
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"🔍 [PARSER] JSON parse failed: {e}, trying fallback methods...")
    
    # Priority 2: Fallback to old format (for backward compatibility)
    # Inside </think> tag (DeepSeek style)
    match = re.search(r'</think>\s*(upregulated|downregulated)', text, re.IGNORECASE | re.DOTALL)
    if match:
        print(f"🔍 [PARSER] Fallback Priority 1 (</think>): '{match.group(1)}'")
        return match.group(1).strip().lower()
    
    # Priority 3: Inside <answer> tag
    match = re.search(r'<answer>(upregulated|downregulated)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        print(f"🔍 [PARSER] Fallback Priority 2 (<answer> tag): '{match.group(1)}'")
        return match.group(1).strip().lower()
    
    # Priority 4: Raw text search (last resort - may pick up from prompt)
    match = re.search(r'(upregulated|downregulated)', text, re.IGNORECASE | re.DOTALL)
    if match:
        print(f"🔍 [PARSER] Fallback Priority 3 (raw text): '{match.group(1)}'")
        print(f"   ⚠️  WARNING: This may be parsing from prompt, not actual output!")
        return match.group(1).strip().lower()
    
    print(f"🔍 [PARSER] No match found, returning 'uncertain'")
    return "uncertain"


def parse_llm_score(text: str) -> int:
    """Extracts an integer score (0-10) from text."""
    import json
    
    # Priority 1: Try to parse as JSON
    try:
        # Find JSON object in text (may be surrounded by other text)
        json_match = re.search(r'\{[^{}]*"overall_score"[^{}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            if "overall_score" in parsed:
                score = int(parsed["overall_score"])
                if 1 <= score <= 10:
                    return score
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    
    # Priority 2: Try to find "overall_score" key-value pair in JSON-like format
    try:
        # Look for "overall_score": <number> pattern
        score_match = re.search(r'"overall_score"\s*:\s*(\d+)', text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            if 1 <= score <= 10:
                return score
    except (ValueError, AttributeError):
        pass
    
    # Priority 3: Fallback to original pattern matching
    matches = re.findall(r"overall\s*score\D*(\d+)", text, re.IGNORECASE | re.DOTALL)
    if matches:
        try:
            score = int(matches[-1])
            if 1 <= score <= 10:
                return score
        except (ValueError, TypeError):
            pass
    
    return -1

