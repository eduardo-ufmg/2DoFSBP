r_out = 0.06 # [m], 60 mm
r_in = 0.05 # [m], 50 mm
mass = 0.05 # [kg], 50 g
# approximate inertia of a hollow cylinder: I = 0.5*m*(r_out^2 + r_in^2)
J = 0.5 * mass * (r_out**2 + r_in**2)  # [kg m^2]
print(f"Using approximate inertia J = {J:.6f} kg m^2")