from ollama import chat

OLLAMA_MODEL = "llama3.2:3b"

def get_temperature(city: str) -> str:
	"""
		Get the current temperature of the city
		args:
			city: the name of the city
		returns:
			The current city temperature
	"""
	city = city.title()
	temperature = {
		'Paris' : '30 °C',
		'Lille': '28 °C',
		'Marseille': '34 °C'
	}
	return temperature.get(city, "Unknown")

messages = [{'role': 'user', 'content': "what is the temperature in Paris"}]
response = chat(
	model= OLLAMA_MODEL,
	tools = [get_temperature],
	messages= messages,
)

if response.message.tool_calls :
	call = response.message.tool_calls[0]
	result = get_temperature(**call.function.arguments)

	messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

	final_response = chat(model=OLLAMA_MODEL, messages=messages, tools=[get_temperature])
	print(final_response.message.content)
