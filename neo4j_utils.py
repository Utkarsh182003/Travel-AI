import os
from neo4j import GraphDatabase, basic_auth
import datetime


from dotenv import load_dotenv
load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_driver = None

def get_neo4j_driver():
    """
    Initializes and returns a singleton Neo4j driver instance.
    """
    global _driver
    if _driver is None:
        if not NEO4J_USERNAME or not NEO4J_PASSWORD:
            print("Error: NEO4J_USERNAME or NEO4J_PASSWORD not found in environment variables. Please set them in your .env file.")
            return None
        try:
            _driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD))
            _driver.verify_connectivity()
            print("Neo4j driver initialized and connected.")
        except Exception as e:
            print(f"Failed to connect to Neo4j. Check URI, credentials, and if Neo4j is running: {e}")
            _driver = None
    return _driver

def close_neo4j_driver():
    """
    Closes the Neo4j driver connection.
    """
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        print("Neo4j driver closed.")

def clear_and_load_sample_data():
    """
    Clears all existing data in the Neo4j database and loads initial sample data
    for airports and cities, ensuring a clean slate for demonstration.
    """
    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j connection not available. Cannot clear and load sample data.")
        return

    try:
        with driver.session() as session:
            print("Clearing existing data from Neo4j...")
            session.run("MATCH (n) DETACH DELETE n")
            print("Data cleared.")

            print("Loading initial sample data (Airports and Cities)...")
            # Create Cities
            session.run("MERGE (:City {name: 'London'})")
            session.run("MERGE (:City {name: 'Paris'})")
            session.run("MERGE (:City {name: 'New York'})")
            session.run("MERGE (:City {name: 'Dubai'})")
            session.run("MERGE (:City {name: 'Delhi'})")
            session.run("MERGE (:City {name: 'Mumbai'})")


            # Creates Airports and link to Cities
            session.run("MERGE (:Airport {name: 'London Heathrow Airport', iata: 'LHR'})-[:LOCATED_IN]->(:City {name: 'London'})")
            session.run("MERGE (:Airport {name: 'Paris Charles de Gaulle Airport', iata: 'CDG'})-[:LOCATED_IN]->(:City {name: 'Paris'})")
            session.run("MERGE (:Airport {name: 'Indira Gandhi International Airport', iata: 'DEL'})-[:LOCATED_IN]->(:City {name: 'Delhi'})")
            session.run("MERGE (:Airport {name: 'Chhatrapati Shivaji Maharaj International Airport', iata: 'BOM'})-[:LOCATED_IN]->(:City {name: 'Mumbai'})")

            print("Initial sample data loaded.")
            load_mock_flights_to_kg()
            load_mock_hotels_to_kg()

    except Exception as e:
        print(f"An error occurred during clearing or loading sample data: {e}")

def query_knowledge_graph(cypher_query: str):
    """
    Executes a Cypher query against the Neo4j knowledge graph and returns the results.
    """
    driver = get_neo4j_driver()
    if not driver:
        return {"status": "error", "message": "Neo4j connection not available. Cannot query knowledge graph."}

    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            data = [record.data() for record in result]
            if data:
                return {"status": "success", "data": data}
            else:
                return {"status": "success", "message": "No results found for the query."}
    except Exception as e:
        return {"status": "error", "message": f"Error executing Cypher query: {e}"}

def add_flight_route_to_kg(origin_iata: str, destination_iata: str, airline_name: str, flight_number: str | None = None, duration_hours: float | None = None):
    """
    Adds a flight route to the knowledge graph.
    Ensures Airport and Airline nodes exist and creates FLIES_TO relationship.
    """
    driver = get_neo4j_driver()
    if not driver:
        return {"status": "error", "message": "Neo4j connection not available. Cannot add flight route."}

    try:
        with driver.session() as session:
            # MERGE airports and airline to ensure they exist
            session.run("MERGE (:Airport {iata: $origin_iata})", origin_iata=origin_iata)
            session.run("MERGE (:Airport {iata: $destination_iata})", destination_iata=destination_iata)
            session.run("MERGE (:Airline {name: $airline_name})", airline_name=airline_name)

            # Create or merge the FLIES_TO relationship
            query = """
            MATCH (origin:Airport {iata: $origin_iata})
            MATCH (destination:Airport {iata: $destination_iata})
            MATCH (airline:Airline {name: $airline_name})
            MERGE (origin)-[r:FLIES_TO]->(destination)
            ON CREATE SET r.airlines = [$airline_name], r.flight_numbers = [$flight_number], r.duration_hours = $duration_hours
            ON MATCH SET r.airlines = CASE WHEN NOT $airline_name IN r.airlines THEN r.airlines + $airline_name ELSE r.airlines END,
                         r.flight_numbers = CASE WHEN NOT $flight_number IS NULL AND NOT $flight_number IN r.flight_numbers THEN r.flight_numbers + $flight_number ELSE r.flight_numbers END
            RETURN origin.iata, destination.iata, r.airlines, r.flight_numbers
            """
            result = session.run(query, origin_iata=origin_iata, destination_iata=destination_iata,
                                 airline_name=airline_name, flight_number=flight_number, duration_hours=duration_hours)
            record = result.single()
            if record:
                return {"status": "success", "message": f"Flight route from {record['origin.iata']} to {record['destination.iata']} added/updated for airlines: {record['r.airlines']}"}
            else:
                return {"status": "error", "message": "Failed to add/update flight route."}
    except Exception as e:
        return {"status": "error", "message": f"Error adding flight route to KG: {e}"}

def add_hotel_to_kg(hotel_name: str, city: str, stars: int, has_pool: bool = False, has_wifi: bool = False):
    """
    Adds a new hotel (or updates an existing one) to the knowledge graph.
    Ensures the city node exists.
    """
    driver = get_neo4j_driver()
    if not driver:
        return {"status": "error", "message": "Neo4j connection not available. Cannot add hotel."}

    try:
        with driver.session() as session:
            # MERGE city to ensure it exists
            session.run("MERGE (:City {name: $city})", city=city)

            # Create or merge the Hotel node and link to city
            query = """
            MATCH (c:City {name: $city})
            MERGE (h:Hotel {name: $hotel_name, city: $city})
            ON CREATE SET h.stars = $stars, h.has_pool = $has_pool, h.has_wifi = $has_wifi
            ON MATCH SET h.stars = $stars, h.has_pool = $has_pool, h.has_wifi = $has_wifi
            MERGE (h)-[:LOCATED_IN]->(c)
            RETURN h.name, h.city, h.stars
            """
            result = session.run(query, hotel_name=hotel_name, city=city, stars=stars, has_pool=has_pool, has_wifi=has_wifi)
            record = result.single()
            if record:
                return {"status": "success", "message": f"Hotel '{record['h.name']}' in {record['h.city']} added/updated."}
            else:
                return {"status": "error", "message": "Failed to add/update hotel."}
    except Exception as e:
        return {"status": "error", "message": f"Error adding hotel to KG: {e}"}

# --- NEW FUNCTIONS FOR MOCK DATA LOADING AND RETRIEVAL FROM KG ---

# Dummy mock flight data (will be loaded into KG)
MOCK_FLIGHTS_DATA = {
    "LHR-CDG-2025-07-20": [
        {"flight_number": "BA303", "airline": "British Airways", "flight_class": "Economy", "price_estimate": 150.00, "departure_airport": "London Heathrow", "departure_iata": "LHR", "scheduled_departure": "2025-07-20 09:00", "arrival_airport": "Paris Charles de Gaulle", "arrival_iata": "CDG", "scheduled_arrival": "2025-07-20 11:30", "status": "scheduled", "layovers": 0, "duration_hours": 2.5},
        {"flight_number": "AF101", "airline": "Air France", "flight_class": "Economy", "price_estimate": 165.00, "departure_airport": "London Heathrow", "departure_iata": "LHR", "scheduled_departure": "2025-07-20 10:30", "arrival_airport": "Paris Charles de Gaulle", "arrival_iata": "CDG", "scheduled_arrival": "2025-07-20 13:00", "status": "scheduled", "layovers": 0, "duration_hours": 2.5},
    ],
    "DEL-BOM-2025-07-25": [
        {"flight_number": "6E202", "airline": "IndiGo", "flight_class": "Economy", "price_estimate": 75.00, "departure_airport": "Indira Gandhi International Airport", "departure_iata": "DEL", "scheduled_departure": "2025-07-25 08:00", "arrival_airport": "Chhatrapati Shivaji Maharaj International Airport", "arrival_iata": "BOM", "scheduled_arrival": "2025-07-25 10:00", "status": "scheduled", "layovers": 0, "duration_hours": 2.0},
        {"flight_number": "AI808", "airline": "Air India", "flight_class": "Economy", "price_estimate": 90.00, "departure_airport": "Indira Gandhi International Airport", "departure_iata": "DEL", "scheduled_departure": "2025-07-25 11:30", "arrival_airport": "Chhatrapati Shivaji Maharaj International Airport", "arrival_iata": "BOM", "scheduled_arrival": "2025-07-25 13:30", "status": "scheduled", "layovers": 0, "duration_hours": 2.0},
    ],
    "BOM-DEL-2025-07-25": [
        {"flight_number": "UK999", "airline": "Vistara", "flight_class": "Economy", "price_estimate": 80.00, "departure_airport": "Chhatrapati Shivaji Maharaj International Airport", "departure_iata": "BOM", "scheduled_departure": "2025-07-25 14:00", "arrival_airport": "Indira Gandhi International Airport", "arrival_iata": "DEL", "scheduled_arrival": "2025-07-25 16:00", "status": "scheduled", "layovers": 0, "duration_hours": 2.0},
    ]
}

# Dummy mock hotel data (will be loaded into KG)
MOCK_HOTELS_DATA = {
    "London": [
        {"id": "HL101", "name": "The Grand London Hotel", "stars": 5, "price_per_night": 250.0, "amenities": ["pool", "wifi", "spa", "gym"], "address": "123 Park Lane"},
        {"id": "HL102", "name": "River View Inn", "stars": 3, "price_per_night": 120.0, "amenities": ["wifi", "breakfast"], "address": "45 River St"},
    ],
    "New York": [
        {"id": "HL103", "name": "Times Square Suites", "stars": 4, "price_per_night": 300.0, "amenities": ["wifi", "gym"], "address": "Broadway"},
        {"id": "HL104", "name": "Central Park Residence", "stars": 5, "price_per_night": 450.0, "amenities": ["pool", "spa", "wifi", "rooftop bar"], "address": "5th Ave"},
    ],
    "Paris": [
        {"id": "HL105", "name": "Eiffel Tower Inn", "stars": 3, "price_per_night": 180.0, "amenities": ["wifi", "cafe"], "address": "Rue Cler"},
        {"id": "HL106", "name": "Le Louvre Grand", "stars": 5, "price_per_night": 500.0, "amenities": ["pool", "spa", "wifi", " Michelin restaurant"], "address": "Near Louvre"},
    ],
    "Dubai": [
        {"id": "HL115", "name": "Burj View Hotel", "stars": 5, "price_per_night": 500.0, "amenities": ["infinity pool", "spa", "wifi", "fine dining", "gym"], "address": "Downtown Dubai"},
        {"id": "HL116", "name": "Desert Oasis Resort", "stars": 4, "price_per_night": 300.0, "amenities": ["pool", "desert safari access", "wifi"], "address": "Outskirts of Dubai"},
        {"id": "HL122", "name": "Palm Jumeirah Resort", "stars": 5, "price_per_night": 600.0, "amenities": ["private beach", "pool", "spa", "water park access", "wifi"], "address": "Palm Jumeirah"},
    ]
}


def load_mock_flights_to_kg():
    """
    Loads mock flight data from MOCK_FLIGHTS_DATA into the Neo4j knowledge graph.
    Creates Flight, Airport, and Airline nodes and relationships.
    """
    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j connection not available. Cannot load mock flights.")
        return

    try:
        with driver.session() as session:
            print("Loading mock flight data into Neo4j KG...")
            for key, flights in MOCK_FLIGHTS_DATA.items():
                for flight_data in flights:
                    origin_iata = flight_data['departure_iata']
                    destination_iata = flight_data['arrival_iata']
                    airline_name = flight_data['airline']
                    flight_number = flight_data['flight_number']
                    flight_date_str = flight_data['scheduled_departure'].split(' ')[0]
                    
                    # Ensure Airport and Airline nodes exist
                    session.run("MERGE (:Airport {iata: $iata, name: $name})", iata=origin_iata, name=flight_data['departure_airport'])
                    session.run("MERGE (:Airport {iata: $iata, name: $name})", iata=destination_iata, name=flight_data['arrival_airport'])
                    session.run("MERGE (:Airline {name: $name})", name=airline_name)

                    # Create/Merge Flight node and its relationships
                    # Use a unique ID for each flight instance based on number, date, origin, dest
                    flight_id = f"{flight_number}-{flight_date_str}-{origin_iata}-{destination_iata}"
                    
                    query = """
                    MATCH (origin:Airport {iata: $origin_iata})
                    MATCH (destination:Airport {iata: $destination_iata})
                    MATCH (airline:Airline {name: $airline_name})
                    MERGE (f:Flight {id: $flight_id})
                    ON CREATE SET f.flight_number = $flight_number, f.date = $flight_date_str,
                                  f.price_estimate = $price_estimate, f.departure_time = $departure_time,
                                  f.arrival_time = $arrival_time, f.duration_hours = $duration_hours,
                                  f.flight_class = $flight_class, f.status = $status, f.layovers = $layovers
                    ON MATCH SET f.price_estimate = $price_estimate, f.status = $status // Update if needed
                    MERGE (origin)-[:DEPARTS_FROM]->(f)
                    MERGE (f)-[:ARRIVES_AT]->(destination)
                    MERGE (f)-[:OPERATED_BY]->(airline)
                    RETURN f.id
                    """
                    session.run(query, 
                                origin_iata=origin_iata, destination_iata=destination_iata,
                                airline_name=airline_name, flight_id=flight_id,
                                flight_number=flight_number, flight_date_str=flight_date_str,
                                price_estimate=flight_data['price_estimate'],
                                departure_time=flight_data['scheduled_departure'],
                                arrival_time=flight_data['scheduled_arrival'],
                                duration_hours=flight_data['duration_hours'],
                                flight_class=flight_data['flight_class'], status=flight_data['status'], layovers=flight_data['layovers']
                                )
            print("Mock flight data loaded into Neo4j KG.")
    except Exception as e:
        print(f"Error loading mock flight data to KG: {e}")

def load_mock_hotels_to_kg():
    """
    Loads mock hotel data from MOCK_HOTELS_DATA into the Neo4j knowledge graph.
    Creates Hotel, City, and Amenity nodes and relationships.
    """
    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j connection not available. Cannot load mock hotels.")
        return

    try:
        with driver.session() as session:
            print("Loading mock hotel data into Neo4j KG...")
            for city_name, hotels in MOCK_HOTELS_DATA.items():
                session.run("MERGE (:City {name: $city_name})", city_name=city_name)
                
                for hotel_data in hotels:
                    hotel_name = hotel_data['name']
                    hotel_id = hotel_data['id']
                    
                    # Create/Merge Hotel node
                    query_hotel = """
                    MATCH (c:City {name: $city_name})
                    MERGE (h:Hotel {id: $hotel_id, name: $hotel_name})
                    ON CREATE SET h.stars = $stars, h.price_per_night = $price_per_night, h.address = $address
                    ON MATCH SET h.stars = $stars, h.price_per_night = $price_per_night, h.address = $address
                    MERGE (h)-[:LOCATED_IN]->(c)
                    RETURN h.id
                    """
                    session.run(query_hotel, city_name=city_name, hotel_id=hotel_id, hotel_name=hotel_name,
                                stars=hotel_data['stars'], price_per_night=hotel_data['price_per_night'],
                                address=hotel_data['address'])

                    for amenity_name in hotel_data['amenities']:
                        query_amenity = """
                        MATCH (h:Hotel {id: $hotel_id})
                        MERGE (a:Amenity {name: $amenity_name})
                        MERGE (h)-[:HAS_AMENITY]->(a)
                        """
                        session.run(query_amenity, hotel_id=hotel_id, amenity_name=amenity_name)
            print("Mock hotel data loaded into Neo4j KG.")
    except Exception as e:
        print(f"Error loading mock hotel data to KG: {e}")

def get_flights_from_kg(origin_iata: str, destination_iata: str, flight_date: str):
    """
    Retrieves flight information from the Neo4j knowledge graph.
    """
    driver = get_neo4j_driver()
    if not driver:
        return {"status": "error", "message": "Neo4j connection not available. Cannot search flights."}

    try:
        with driver.session() as session:
            # Cypher query to find flights
            query = """
            MATCH (origin:Airport {iata: $origin_iata})-[:DEPARTS_FROM]->(flight:Flight {date: $flight_date})-[:ARRIVES_AT]->(destination:Airport {iata: $destination_iata})
            MATCH (flight)-[:OPERATED_BY]->(airline:Airline)
            RETURN 
                flight.flight_number AS flight_number,
                airline.name AS airline,
                flight.flight_class AS flight_class,
                flight.price_estimate AS price_estimate,
                origin.name AS departure_airport,
                origin.iata AS departure_iata,
                flight.departure_time AS scheduled_departure,
                destination.name AS arrival_airport,
                destination.iata AS arrival_iata,
                flight.arrival_time AS scheduled_arrival,
                flight.status AS status,
                flight.layovers AS layovers,
                flight.duration_hours AS duration_hours
            ORDER BY flight.price_estimate
            """
            result = session.run(query, origin_iata=origin_iata, destination_iata=destination_iata, flight_date=flight_date)
            flights = [record.data() for record in result]

            if flights:
                return {"status": "success", "flights": flights}
            else:
                return {"status": "success", "message": "No flights found for the given criteria."}
    except Exception as e:
        return {"status": "error", "message": f"Error searching flights in KG: {e}"}

def get_hotels_from_kg(city: str, amenities: list[str] = []):
    """
    Retrieves hotel information from the Neo4j knowledge graph based on city and amenities.
    """
    driver = get_neo4j_driver()
    if not driver:
        return {"status": "error", "message": "Neo4j connection not available. Cannot search hotels."}

    try:
        with driver.session() as session:
            query = """
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
            """
            if amenities:
                amenity_match_clauses = [f"(:Amenity {{name: '{amenity.strip()}'}})" for amenity in amenities]
                query += f" MATCH (h)-[:HAS_AMENITY]->{', (h)-[:HAS_AMENITY]->'.join(amenity_match_clauses)}"
            
            query += """
            OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
            RETURN 
                h.id AS id,
                h.name AS name,
                h.stars AS stars,
                h.price_per_night AS price_per_night,
                COLLECT(a.name) AS amenities,
                h.address AS address,
                c.name AS city
            ORDER BY h.stars DESC, h.price_per_night ASC
            """
            
            result = session.run(query, city=city)
            hotels = [record.data() for record in result]

            if hotels:
                return {"status": "success", "hotels": hotels}
            else:
                return {"status": "success", "message": f"No hotels found in {city} matching the criteria."}
    except Exception as e:
        return {"status": "error", "message": f"Error searching hotels in KG: {e}"}

