"""
Generate animation (GIF or MP4) from PLY simulation files using Open3D.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
import imageio
from typing import Optional, Tuple
import argparse
import os
import sys
import warnings
from tqdm import tqdm

# Suppress Open3D warnings and engine output
warnings.filterwarnings('ignore')
os.environ['OPEN3D_DISABLE_CONSOLE_WARNING'] = '1'

# Set Open3D verbosity to suppress info messages
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class SuppressOutput:
    """Context manager to suppress stdout and stderr at the file descriptor level."""
    def __enter__(self):
        # Save original file descriptors
        self._original_stdout_fd = os.dup(1)
        self._original_stderr_fd = os.dup(2)

        # Open devnull
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Redirect stdout and stderr to devnull
        os.dup2(self._devnull_fd, 1)
        os.dup2(self._devnull_fd, 2)

        # Also redirect Python-level streams
        sys.stdout = os.fdopen(self._original_stdout_fd, 'w')
        sys.stderr = os.fdopen(self._original_stderr_fd, 'w')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original file descriptors
        os.dup2(self._original_stdout_fd, 1)
        os.dup2(self._original_stderr_fd, 2)

        # Close duplicates and devnull
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
        os.close(self._devnull_fd)

        # Restore Python-level streams
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def render_point_cloud_to_image_offscreen(
    pcd: o3d.geometry.PointCloud,
    width: int = 1920,
    height: int = 1080,
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    front: Tuple[float, float, float] = (0.0, -1.0, -0.5),
    lookat: Optional[Tuple[float, float, float]] = None,
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    zoom: float = 0.8,
    show_axes: bool = False,
    axes_size: float = 0.1,
    show_grid: bool = False,
    grid_size: float = 1.0,
    grid_divisions: int = 10
) -> np.ndarray:
    """Render a point cloud to an image using offscreen rendering."""
    # Suppress rendering engine warnings
    with SuppressOutput():
        # Create offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # Setup material for point cloud
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = point_size

        # Add point cloud
        renderer.scene.add_geometry("pcd", pcd, material)

        # Add coordinate frame (3D axes) if requested
        if show_axes:
            # Create coordinate frame at origin
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axes_size, origin=[0, 0, 0]
            )
            axes_material = o3d.visualization.rendering.MaterialRecord()
            axes_material.shader = "defaultUnlit"
            renderer.scene.add_geometry("axes", coord_frame, axes_material)

        # Add grid if requested
        if show_grid:
            # Create grid on XY plane at Z=0
            grid_lines = []
            half_size = grid_size / 2
            step = grid_size / grid_divisions

            # Lines parallel to X-axis
            for i in range(grid_divisions + 1):
                y = -half_size + i * step
                grid_lines.append([[-half_size, y, 0], [half_size, y, 0]])

            # Lines parallel to Y-axis
            for i in range(grid_divisions + 1):
                x = -half_size + i * step
                grid_lines.append([[x, -half_size, 0], [x, half_size, 0]])

            # Create LineSet
            points = []
            lines = []
            for i, (start, end) in enumerate(grid_lines):
                points.append(start)
                points.append(end)
                lines.append([2*i, 2*i+1])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            # Set grid color (light gray)
            colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)

            grid_material = o3d.visualization.rendering.MaterialRecord()
            grid_material.shader = "unlitLine"
            grid_material.line_width = 1.0
            renderer.scene.add_geometry("grid", line_set, grid_material)

        # Set background color (needs RGBA format)
        bg_rgba = np.array([*background_color, 1.0], dtype=np.float32)
        renderer.scene.set_background(bg_rgba)

        # Setup camera
        center = pcd.get_center() if lookat is None else np.array(lookat)
        eye = center + np.array(front) * 2.0 / zoom
        renderer.setup_camera(60.0, center, eye, up)

        # Render
        image = renderer.render_to_image()
        image_np = np.asarray(image)

    return image_np


def generate_animation(
    input_dir: str,
    output_file: str,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    zoom: float = 0.8,
    quality: int = 8,
    front: Tuple[float, float, float] = (0.0, -1.0, -0.5),
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    show_axes: bool = False,
    axes_size: float = 0.1,
    show_grid: bool = False,
    grid_size: float = 1.0,
    grid_divisions: int = 10
):
    """
    Generate animation from PLY files.

    Args:
        input_dir: Directory containing PLY files
        output_file: Output animation file (e.g., 'animation.mp4' or 'animation.gif')
        fps: Frames per second for the animation
        width: Image width in pixels (will be adjusted to be divisible by 16 for MP4)
        height: Image height in pixels (will be adjusted to be divisible by 16 for MP4)
        point_size: Size of points in rendering
        background_color: RGB background color (0-1 range)
        zoom: Camera zoom level
        quality: Quality for MP4 (1-10, higher is better) or GIF compression
        front: Camera viewing direction
        up: Camera up vector
        show_axes: Whether to show 3D coordinate axes (X=red, Y=green, Z=blue)
        axes_size: Size of the coordinate axes
        show_grid: Whether to show ground grid on XY plane at Z=0
        grid_size: Size of the grid
        grid_divisions: Number of divisions in the grid
    """
    input_path = Path(input_dir)

    # Find all PLY files
    ply_files = sorted(input_path.glob("*.ply"))

    if not ply_files:
        raise ValueError(f"No PLY files found in {input_dir}")

    # Adjust dimensions to be divisible by 16 for MP4 compatibility
    output_path = Path(output_file)
    if output_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        original_width, original_height = width, height
        width = ((width + 15) // 16) * 16
        height = ((height + 15) // 16) * 16
        if width != original_width or height != original_height:
            print(f"Adjusted resolution from {original_width}x{original_height} to {width}x{height} (divisible by 16)")

    print(f"Found {len(ply_files)} PLY files")
    print(f"Generating animation: {output_file}")

    # Load first point cloud to determine camera lookat point
    first_pcd = o3d.io.read_point_cloud(str(ply_files[0]))
    lookat = first_pcd.get_center()

    # Render all frames with progress bar
    frames = []
    for ply_file in tqdm(ply_files, desc="Rendering frames", unit="frame"):
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(ply_file))

        # Color the point cloud if it doesn't have colors
        if not pcd.has_colors():
            points = np.asarray(pcd.points)
            z_coords = points[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_norm = (z_coords - z_min) / (z_max - z_min + 1e-8)

            # Create color gradient (blue to red based on height)
            colors = np.zeros((len(points), 3))
            colors[:, 0] = z_norm  # Red increases with height
            colors[:, 2] = 1 - z_norm  # Blue decreases with height
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Render frame using offscreen rendering
        image = render_point_cloud_to_image_offscreen(
            pcd,
            width=width,
            height=height,
            point_size=point_size,
            background_color=background_color,
            front=front,
            lookat=lookat,
            up=up,
            zoom=zoom,
            show_axes=show_axes,
            axes_size=axes_size,
            show_grid=show_grid,
            grid_size=grid_size,
            grid_divisions=grid_divisions
        )
        frames.append(image)

    # Save animation
    output_path = Path(output_file)
    if output_path.suffix.lower() == '.gif':
        print(f"Saving as GIF...")
        imageio.mimsave(output_file, frames, fps=fps, loop=0)
    elif output_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        print(f"Saving as {output_path.suffix.upper()}...")

        # Suppress imageio warnings about macro block size
        import logging
        imageio_logger = logging.getLogger('imageio_ffmpeg')
        old_level = imageio_logger.level
        imageio_logger.setLevel(logging.ERROR)

        try:
            writer = imageio.get_writer(
                output_file,
                fps=fps,
                quality=quality,
                codec='libx264' if output_path.suffix.lower() == '.mp4' else None,
                pixelformat='yuv420p'
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        finally:
            imageio_logger.setLevel(old_level)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    print(f"Animation saved to: {output_file}")
    print(f"Total frames: {len(frames)}")
    print(f"Duration: {len(frames)/fps:.2f} seconds")

    # Clean up frames to reduce memory before exit
    del frames
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Generate animation from PLY files",
        epilog="Note: Some harmless cleanup warnings may appear at the end. "
               "To suppress all warnings, run: your_command 2>/dev/null"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./sim_results/sand',
        help='Directory containing PLY files (default: ./sim_results/sand)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='animation.mp4',
        help='Output file name (e.g., animation.mp4 or animation.gif) (default: animation.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1920,
        help='Image width in pixels (default: 1920)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Image height in pixels (default: 1080)'
    )
    parser.add_argument(
        '--point-size',
        type=float,
        default=2.0,
        help='Point size for rendering (default: 2.0)'
    )
    parser.add_argument(
        '--zoom',
        type=float,
        default=0.8,
        help='Camera zoom level (default: 0.8)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=8,
        help='Video quality 1-10, higher is better (default: 8)'
    )
    parser.add_argument(
        '--bg-color',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help='Background color RGB (0-1 range) (default: 1.0 1.0 1.0 for white)'
    )
    parser.add_argument(
        '--show-axes',
        action='store_true',
        help='Show 3D coordinate axes (X=red, Y=green, Z=blue)'
    )
    parser.add_argument(
        '--axes-size',
        type=float,
        default=0.1,
        help='Size of coordinate axes (default: 0.1)'
    )
    parser.add_argument(
        '--show-grid',
        action='store_true',
        help='Show ground grid on XY plane at Z=0'
    )
    parser.add_argument(
        '--grid-size',
        type=float,
        default=1.0,
        help='Size of grid (default: 1.0)'
    )
    parser.add_argument(
        '--grid-divisions',
        type=int,
        default=10,
        help='Number of grid divisions (default: 10)'
    )

    args = parser.parse_args()

    generate_animation(
        input_dir=args.input_dir,
        output_file=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
        background_color=tuple(args.bg_color),
        zoom=args.zoom,
        quality=args.quality,
        front=(0.0, 1.0, 0.5),
        up=(0.0, 0.0, 1.0),
        show_axes=args.show_axes,
        axes_size=args.axes_size,
        show_grid=args.show_grid,
        grid_size=args.grid_size,
        grid_divisions=args.grid_divisions
    )


if __name__ == "__main__":
    main()

    # Suppress cleanup warnings that occur after main() exits
    # by redirecting stderr during final garbage collection
    import gc
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    gc.collect()
    os.close(devnull)
    os.dup2(saved_stderr, 2)
    os.close(saved_stderr)
