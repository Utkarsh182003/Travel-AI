# 🌍 Travel Booking AI

Travel Booking AI is a knowledge graph-powered application designed to assist users in finding flights and hotels based on their preferences. It leverages Neo4j for managing a knowledge graph, LangChain for document processing, and ChromaDB for vector-based retrieval. This project demonstrates the integration of AI, graph databases, and vector stores for travel-related use cases.

---

## ✨ Features

- **Flight Search**: Retrieve flight information based on origin, destination, and date.
- **Hotel Search**: Retrieve hotel information based on city and amenities.
- **Knowledge Graph (Neo4j)**: Stores and queries relationships between cities, airports, airlines, flights, hotels, and amenities.
- **Vector Search (ChromaDB)**: Uses vector-based retrieval for document-based question answering.
- **Mock Data**: Includes mock data for flights and hotels to demonstrate functionality.
- **Extensibility**: Easily extendable to include more data sources and features.

---

## 🧠 How It Works

1. **Data Loading**: Mock flight and hotel data are loaded into the Neo4j knowledge graph.
2. **Flight Search**: Users can query flights based on origin, destination, and date.
3. **Hotel Search**: Users can query hotels based on city and desired amenities.
4. **Knowledge Graph Queries**: Cypher queries are used to retrieve structured data from Neo4j.
5. **Dynamic Updates**: The knowledge graph can be updated with new flights, hotels, and relationships.

---

## 🚀 Getting Started

Follow these steps to set up and run the Travel Booking AI on your local machine.

### Prerequisites

- Python 3.9+
- Neo4j (Community or Enterprise Edition)

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd travel-booking-ai
```

### 2. Set Up Your Environment

It’s highly recommended to use a virtual environment.

```sh
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Configure Neo4j

Create a file named `.env` in the root directory and add your keys:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

Replace the placeholder values with your actual API keys.

### 5. Run the Application
Backend
```sh
uvicorn main:app
```
Frontend
```sh
streamlit run app.py
```

Your browser should automatically open to the Streamlit application. If not, open your web browser and go to [http://localhost:8501](http://localhost:8501).

---

## 💡 Potential Future Enhancements

- **Real-time Booking Integration:** Connect to flight, hotel, and activity booking APIs.
- **Advanced Knowledge Graph:** Use a persistent graph database for larger datasets.
- **User Profiles & History:** Implement user accounts to save and retrieve past itineraries and learn user preferences over time.
- **Cost Estimation:** Provide estimated costs for activities, transport, and accommodation.
- **Local Event Integration:** Dynamically pull local events and festivals for the travel dates.

---

## 🤝 Contributing

Feel free to fork this repository, open issues, or submit pull requests.

---

## 📄 License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Made with ❤️ by UTKARSH