
import re

def validate_html(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    stack = []
    errors = []
    
    # Simple regex for tags (ignores comments and strings mostly)
    # This is a rough validator for structural integrity
    
    for i, line in enumerate(lines):
        # Remove comments
        line = re.sub(r'<!--.*?-->', '', line)
        
        # Find all tags
        tags = re.findall(r'(</?\w+)', line)
        
        for tag_str in tags:
            is_close = tag_str.startswith('</')
            tag_name = tag_str.replace('</', '').replace('<', '')
            
            if tag_name not in ['div', 'main', 'header', 'aside', 'section', 'article', 'nav', 'script']:
                continue
                
            if not is_close:
                stack.append((tag_name, i + 1))
            else:
                if not stack:
                    errors.append(f"Line {i+1}: Unexpected closing </{tag_name}>")
                else:
                    last_tag, last_line = stack.pop()
                    if last_tag != tag_name:
                         # Allow mismatch if it's typical HTML messiness, but flag it
                         errors.append(f"Line {i+1}: Closing </{tag_name}> but expected </{last_tag}> (opened at {last_line})")

    if stack:
        for tag, line in stack:
            errors.append(f"Line {line}: Unclosed <{tag}>")
            
    return errors

if __name__ == "__main__":
    errors = validate_html('/Users/mac/Downloads/rox3.0/app/templates/index_rox2.html')
    if errors:
        print("Found HTML Structure Errors:")
        for e in errors:
            print(e)
    else:
        print("No structural errors found in major containers.")
