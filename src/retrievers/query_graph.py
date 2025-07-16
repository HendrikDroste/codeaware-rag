from neo4j import GraphDatabase

# Prototype for querying references to a function in a Neo4j graph database
# This file is ony for testing purposes
def query_references(function_name):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    session = driver.session()
    # Cypher to find CALLS edges to the function
    result = session.run(
        "MATCH (caller:Function)-[r:CALLS]->(target:Function {name:$name}) "
        "MATCH (caller)-[:DEFINED_IN]->(file:File) "
        "RETURN file.path AS file, r.line AS line",
        name=function_name
    )
    locations = [f"{record['file']}:{record['line']}" for record in result]
    if locations:
        print(f"{function_name} is referenced in: " + ", ".join(locations))
    else:
        print(f"No references found for function {function_name}.")
    session.close()
    driver.close()

if __name__ == "__main__":
    query_references("send_static_file")
