import numpy as np
import plotly.graph_objects as go
from phys import * # Assuming 'phys' is a custom library as in the original
import astropy.units as u

# --- Simulation Setup (same as original) ---
G = 10. # Artificially small G for slower, stable chaos in small space

particles = [
    Particle(
        mass=1.0,
        position=np.array([-5.0, 0.0, 0.0]),
        velocity=np.array([-0.18, -0.5, -0.25]),
        charge=0.0,
    ),
    Particle(
        mass=1.0,
        position=np.array([5.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.5387, 0.2]),
        charge=0.0,
    ),
    Particle(
        mass=1.0,
        position=np.array([0.0, 8.66, 0.0]),  # ~sqrt(3) * 5, top of equilateral triangle
        velocity=np.array([0.18, 0.1, 0.5]),
        charge=0.0,
    ),
]


sim = Simulation(
    timestep = 0.1,
    particles=particles,
    engines=[Gravity(G)]
)

sim.simulate(end_time=100.0)

data = sim.data

# --- Plotting with Plotly ---

# Create a new Plotly Figure
fig = go.Figure()

# Iterate through each particle's trajectory data
for column in data.get_columns():
    if not column.name.startswith("Particle "):
        continue
    # Extract x, y, and z coordinates from the position data
    x = [position[0] for position in column]
    y = [position[1] for position in column]
    z = [position[2] for position in column]

    # Add a 3D scatter trace to the figure for the current particle
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers', # Show both the path and the points
        name=column.name,     # Set the name for the legend (like 'label')
        marker=dict(
            size=3,           # Corresponds to 'markersize'
            opacity=0.8
        ),
        line=dict(
            width=4           # Corresponds to 'linestyle' (solid is default)
        )
    ))

# Update the layout to add titles and axis labels
fig.update_layout(
    title="3D Particle Trajectories",
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis' # Added Z-axis label for clarity
    ),
    margin=dict(l=0, r=0, b=0, t=40) # Adjust plot margins
)

# Display the interactive plot
fig.show()
