import re

with open('multi_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all mojibake patterns and fix them
# The pattern is: UTF-8 bytes interpreted as latin-1 then re-encoded as UTF-8
def fix_mojibake(text):
    # Try to fix sections that look like mojibake
    result = []
    i = 0
    while i < len(text):
        # Check if we're at a mojibake sequence (starts with \xc3 or \xc2 pattern)
        chunk = text[i:i+200]
        try:
            fixed = chunk.encode('latin-1').decode('utf-8')
            # If this worked and produced Chinese, use it
            if any('\u4e00' <= c <= '\u9fff' for c in fixed[:20]):
                # Find the longest fixable chunk
                for end in range(min(200, len(text)-i), 0, -1):
                    try:
                        fixed = text[i:i+end].encode('latin-1').decode('utf-8')
                        result.append(fixed)
                        i += end
                        break
                    except:
                        continue
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        except:
            result.append(text[i])
            i += 1
    return ''.join(result)

fixed = fix_mojibake(content)

with open('multi_agent.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

# Verify
with open('multi_agent.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 640 <= i <= 660:
            print(f'{i}: {line}', end='')
