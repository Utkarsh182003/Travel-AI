import datetime
from typing import List, Dict, Any
from neo4j_utils import get_flights_from_kg, get_hotels_from_kg 

def simulate_flight_search(origin_iata: str, destination_iata: str, flight_date: str) -> Dict[str, Any]:
    """
    Simulates searching for flights. Now fetches data from the Neo4j Knowledge Graph.
    """
    print(f"Searching for flights from {origin_iata} to {destination_iata} on {flight_date} in Neo4j KG.")

    # Call the new function to get flights from Neo4j
    result = get_flights_from_kg(origin_iata, destination_iata, flight_date)

    if result["status"] == "success":
        flights = result.get("flights", [])
        if flights:
            for flight in flights:
                flight["flight_date"] = flight_date
                flight["price"] = flight.get("price_estimate", 0.0)
            print(f"Found {len(flights)} flights from Neo4j.")
            return {"status": "success", "flights": flights}
        else:
            return {"status": "success", "message": "No flights found for the given criteria (from KG)."}
    else:
        return result 


def simulate_google_hotels(city: str, check_in_date: str, check_out_date: str, num_guests: int = 1, amenities: List[str] = None) -> Dict[str, Any]:
    """
    Simulates searching for hotels. Now fetches data from the Neo4j Knowledge Graph.
    """
    print(f"Searching for hotels in {city} for {num_guests} guests from {check_in_date} to {check_out_date} with amenities {amenities} in Neo4j KG.")

    result = get_hotels_from_kg(city, amenities if amenities is not None else [])

    if result["status"] == "success":
        hotels = result.get("hotels", [])
        if hotels:
            for hotel in hotels:
                hotel["check_in_date"] = check_in_date
                hotel["check_out_date"] = check_out_date
                hotel["num_guests"] = num_guests
            print(f"Found {len(hotels)} hotels from Neo4j.")
            return {"status": "success", "hotels": hotels}
        else:
            return {"status": "success", "message": "No hotels found for the given criteria (from KG)."}
    else:
        return result

