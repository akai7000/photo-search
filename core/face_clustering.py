import numpy as np
from sklearn.cluster import DBSCAN
from core.database import get_connection

def cluster_faces(eps: float = 0.4, min_samples: int = 2):
    """
    Cluster face embeddings using DBSCAN.
    Updates cluster IDs for faces that have not yet been named.
    Returns the number of clusters found.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Load all face embeddings
    cursor.execute("SELECT id, embedding FROM faces WHERE embedding IS NOT NULL AND person_id IS NULL")
    rows = cursor.fetchall()

    if len(rows) < 2:
        print("Not enough faces to cluster.")
        conn.close()
        return 0

    face_ids = []
    embeddings = []

    for row in rows:
        face_ids.append(row["id"])
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    # Cluster
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    # Assign cluster IDs
    # Note: DBSCAN uses -1 for noise/outliers
    for face_id, label in zip(face_ids, labels):
        cluster_id = int(label) if label != -1 else None
        cursor.execute(
            "UPDATE faces SET cluster_id = ? WHERE id = ?",
            (cluster_id, face_id)
        )

    conn.commit()
    conn.close()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clustering complete. Found {n_clusters} people (clusters).")
    return n_clusters