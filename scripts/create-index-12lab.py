from twelvelabs import TwelveLabs
client = TwelveLabs(api_key="<api-key>")
engines = [
        {
            "name": "pegasus1.1",
            "options": ["visual", "conversation"]
        }
    ]
index = client.index.create(
    name="Checker2",
    engines=engines,
    addons=["thumbnail"] # Optional
)
print(f"A new index has been created: id={index.id} name={index.name} engines={index.engines}")

# Store the index name, ID and engine name in a txt file
with open("12lab-index.txt", "w") as file:
    file.write(f"A new index has been created: id={index.id} name={index.name} engines={index.engines}\n")

