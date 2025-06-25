# neo4j_utils.py
import os
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Neo4j credentials from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") # Ensure you set a strong password in .env

# Global driver instance
driver = None

def get_neo4j_driver():
    """
    Returns a Neo4j driver instance. Initializes it if it doesn't exist.
    """
    global driver
    if driver is None:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD))
            driver.verify_connectivity()
            print("Successfully connected to Neo4j database.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            driver = None # Reset driver if connection fails
    return driver

def close_neo4j_driver():
    """
    Closes the Neo4j driver connection.
    """
    global driver
    if driver is not None:
        driver.close()
        driver = None
        print("Neo4j driver closed.")

def clear_and_load_sample_data():
    """
    Clears existing data and loads sample flight/hotel/user data into Neo4j.
    This function demonstrates populating the graph.
    """
    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j driver not available. Cannot load sample data.")
        return

    with driver.session() as session:
        # Clear existing data to ensure a clean slate for demo
        session.run("MATCH (n) DETACH DELETE n")
        print("Cleared existing Neo4j data.")

        # Create Cities
        session.run("MERGE (:City {name: 'London'})")
        session.run("MERGE (:City {name: 'Paris'})")
        session.run("MERGE (:City {name: 'New York'})")
        session.run("MERGE (:City {name: 'Delhi'})")
        session.run("MERGE (:City {name: 'Mumbai'})")

        # Create Airports
        session.run("MERGE (:Airport {code: 'LHR', name: 'London Heathrow Airport'})-[:LOCATED_IN]->(:City {name: 'London'})")
        session.run("MERGE (:Airport {code: 'CDG', name: 'Charles de Gaulle Airport'})-[:LOCATED_IN]->(:City {name: 'Paris'})")
        session.run("MERGE (:Airport {code: 'JFK', name: 'John F. Kennedy Airport'})-[:LOCATED_IN]->(:City {name: 'New York'})")
        session.run("MERGE (:Airport {code: 'DEL', name: 'Indira Gandhi International Airport'})-[:LOCATED_IN]->(:City {name: 'Delhi'})")
        session.run("MERGE (:Airport {code: 'BOM', name: 'Chhatrapati Shivaji Maharaj International Airport'})-[:LOCATED_IN]->(:City {name: 'Mumbai'})")

        # Create Airlines
        session.run("MERGE (:Airline {name: 'British Airways'})")
        session.run("MERGE (:Airline {name: 'Air France'})")
        session.run("MERGE (:Airline {name: 'Indigo'})")
        session.run("MERGE (:Airline {name: 'Vistara'})")

        # Create Sample Flights (simplified, real data would be more complex)
        session.run("MERGE (a:Airport {code: 'LHR'}) MERGE (b:Airport {code: 'CDG'}) MERGE (c:Airline {name: 'British Airways'}) MERGE (a)-[:HAS_FLIGHT {flight_number: 'BA303', price: 150, duration_hours: 1.5, available_date: '2025-07-20'}]->(b) MERGE (c)-[:OPERATES]->(a)-[:HAS_FLIGHT]->(b)")
        session.run("MERGE (a:Airport {code: 'DEL'}) MERGE (b:Airport {code: 'BOM'}) MERGE (c:Airline {name: 'Indigo'}) MERGE (a)-[:HAS_FLIGHT {flight_number: '6E 202', price: 80, duration_hours: 2, available_date: '2025-07-25'}]->(b) MERGE (c)-[:OPERATES]->(a)-[:HAS_FLIGHT]->(b)")
        session.run("MERGE (a:Airport {code: 'JFK'}) MERGE (b:Airport {code: 'LHR'}) MERGE (c:Airline {name: 'British Airways'}) MERGE (a)-[:HAS_FLIGHT {flight_number: 'BA178', price: 600, duration_hours: 7, available_date: '2025-08-10'}]->(b) MERGE (c)-[:OPERATES]->(a)-[:HAS_FLIGHT]->(b)")

        # Create Sample Hotels
        session.run("MERGE (:Hotel {name: 'The Grand London', city: 'London', stars: 5, price_per_night: 200, has_pool: true, has_spa: true})")
        session.run("MERGE (:Hotel {name: 'Paris Inn', city: 'Paris', stars: 4, price_per_night: 120, has_pool: false, has_spa: false})")
        session.run("MERGE (:Hotel {name: 'NY Central Hotel', city: 'New York', stars: 3, price_per_night: 180, has_pool: true, has_spa: false})")
        session.run("MERGE (:Hotel {name: 'Taj Palace Delhi', city: 'Delhi', stars: 5, price_per_night: 250, has_pool: true, has_spa: true})")

        # Create Users and their preferences/bookings (example)
        session.run("MERGE (:User {name: 'Alice'})-[:PREFERS_AIRLINE]->(:Airline {name: 'British Airways'})")
        session.run("MERGE (:User {name: 'Bob'})-[:PREFERS_HOTEL_AMENITY {amenity: 'Pool'}]->(:City {name: 'New York'})")

        print("Sample data loaded into Neo4j.")

def query_knowledge_graph(cypher_query: str):
    """
    Executes a Cypher query against the Neo4j database and returns results.
    """
    driver = get_neo4j_driver()
    if not driver:
        return {"status": "error", "message": "Neo4j connection not available."}

    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
            print(f"Executed Cypher query: {cypher_query}")
            print(f"Knowledge Graph query results: {records}")
            return {"status": "success", "data": records}
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
        return {"status": "error", "message": f"Error querying Knowledge Graph: {e}"}

# Ensure driver is closed when the script exits
import atexit
atexit.register(close_neo4j_driver)
