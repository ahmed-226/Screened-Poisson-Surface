import argparse
import sys
import open3d as o3d
import os

from src.preprocess import load_point_cloud, estimate_normals
from src.reconstruction import run_poisson
from src.postprocess import clean_mesh, filter_by_density
from src.utils import generate_sphere_point_cloud, visualize

def main():
    parser = argparse.ArgumentParser(description="Screened Poisson Surface Reconstruction")
    parser.add_argument("--input", type=str, help="Path to input point cloud (PLY, XYZ, etc.). If not provided, a sphere is generated.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output mesh. If not provided, auto-generated in data/.")
    parser.add_argument("--depth", type=int, default=8, help="Octree depth (resolution).")
    parser.add_argument("--scale", type=float, default=1.1, help="Bounding box scale.")
    parser.add_argument("--density_quantile", type=float, default=0.01, help="Quantile of low-density vertices to trim.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the result.")
    parser.add_argument("--manual", action="store_true", help="Use manual Python implementation (slower, demonstrative).")
    parser.add_argument("--decimate", type=int, default=None, help="Target number of triangles for mesh decimation.")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        os.makedirs("data", exist_ok=True)
        method = "manual" if args.manual else "open3d"
        input_name = os.path.splitext(os.path.basename(args.input))[0] if args.input else "sphere"
        output_path = f"data/{input_name}_{method}_d{args.depth}.ply"
    else:
        output_path = args.output
    
    # 1. Input
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)
        pcd = load_point_cloud(args.input)
    else:
        print("No input provided. Generating synthetic sphere...")
        pcd = generate_sphere_point_cloud(noise_std=0.02)
        
    # 2. Preprocess
    pcd = estimate_normals(pcd)
    
    # 3. Reconstruction
    if args.manual:
        from src.reconstruction import run_poisson_manual
        mesh, densities = run_poisson_manual(pcd, depth=args.depth, scale=args.scale)
    else:
        mesh, densities = run_poisson(pcd, depth=args.depth, scale=args.scale)
    
    # 4. Post-process
    mesh = clean_mesh(mesh)
    
    # 5. Decimate (optional)
    if args.decimate and len(mesh.triangles) > args.decimate:
        print(f"Decimating mesh from {len(mesh.triangles)} to {args.decimate} triangles...")
        mesh = mesh.simplify_quadric_decimation(args.decimate)
        mesh.compute_vertex_normals()
        print(f"After decimation: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
    
    # 5. Save
    print(f"Saving mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    # 6. Visualize
    if args.visualize:
        print("Visualizing (Input Point Cloud + Reconstructed Mesh)...")
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        mesh_translated = o3d.geometry.TriangleMesh(mesh)
        mesh_translated.translate((extent[0] * 1.5, 0, 0))
        
        visualize([pcd, mesh_translated], window_name="Result")

if __name__ == "__main__":
    main()
