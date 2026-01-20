## Core Steps of DBSCAN

### 1. Choose the Two Parameters
- **ε (epsilon):** Radius of the neighborhood around a point  
- **minPts:** Minimum number of points required to form a dense region  

These two values define what “dense” means for your dataset.

---

### 2. Classify Each Point
For every point in the dataset:

- Count how many points fall within its **ε‑radius**.
- Label each point as:
  - **Core point:** at least `minPts` neighbors  
  - **Border point:** fewer than `minPts` neighbors but inside a core point’s neighborhood  
  - **Noise point:** neither core nor border  

---

### 3. Form Clusters
- Pick an unvisited point.
- If it’s a **core point**, start a new cluster:
  - Add all points in its ε‑neighborhood.
  - For every neighbor that is also a core point, expand the cluster by adding its neighbors.
  - Continue until no more points can be added.
- If it’s **not** a core point, mark it as noise (unless it later becomes a border point of a cluster).

---

### 4. Repeat Until All Points Are Visited
You end up with:

- One or more clusters of connected dense regions  
- Some noise/outlier points  

---

## Why DBSCAN Is Popular
- Finds arbitrarily shaped clusters  
- Automatically detects outliers  
- No need to specify the number of clusters  
