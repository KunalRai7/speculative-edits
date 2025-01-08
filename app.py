from flask import Flask, request, jsonify, render_template
from main import vanilla_edit, speculative_edit, get_readme_prompt

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_readme_prompt')
def readme_prompt():
    return jsonify({'prompt': get_readme_prompt()})

@app.route('/edit', methods=['POST'])
def edit():
    data = request.get_json()
    method = data.get('method', 'speculative')
    # Always use README prompt as per requirements
    prompt = get_readme_prompt()
    max_tokens = int(data.get('max_tokens', 1000))
    
    try:
        if method == 'vanilla':
            result = vanilla_edit(prompt, max_tokens)
        else:
            result = speculative_edit(prompt, max_tokens)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 