from flask import Flask, request, jsonify
from main import vanilla_edit, speculative_edit, get_readme_prompt

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>Speculative Edits API</h1>
    <p>Use /edit endpoint with POST request to edit code.</p>
    <p>Parameters:</p>
    <ul>
        <li>method: "vanilla" or "speculative"</li>
        <li>prompt: (optional) custom prompt, or will use README example</li>
        <li>max_tokens: (optional) maximum tokens to generate, default 1000</li>
    </ul>
    '''

@app.route('/edit', methods=['POST'])
def edit():
    data = request.get_json()
    method = data.get('method', 'speculative')
    prompt = data.get('prompt', get_readme_prompt())
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