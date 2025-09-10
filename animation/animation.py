import imageio

# Simple conversion - every 3rd frame for reasonable file size
reader = imageio.get_reader('/mnt/data/lk1167/projects/AngleCamV2/animation/maranta_leuconeura_timelapse.mp4')
frames = []

for i, frame in enumerate(reader):
    if i % 3 == 0:  # Every 3rd frame
        frames.append(frame)

# Save as GIF
imageio.mimsave('/mnt/data/lk1167/projects/AngleCamV2/animation/maranta_leuconeura_timelapse.gif', 
                frames, duration=0.2, loop=0)

print(f'GIF created with {len(frames)} frames!')