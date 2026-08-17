import sys

with open('crates/daemon-core/src/http.rs', 'r') as f:
    content = f.read()

# Fix the missing };
old = 'image_input: extract_image_input(&payload.messages),\n\n    {'
new = 'image_input: extract_image_input(&payload.messages),\n    };\n\n    {'

count = content.count(old)
print('Missing }; occurrences:', count)

if count == 1:
    content = content.replace(old, new, 1)
    with open('crates/daemon-core/src/http.rs', 'w') as f:
        f.write(content)
    print('Fixed')
else:
    print('ERROR: count != 1')