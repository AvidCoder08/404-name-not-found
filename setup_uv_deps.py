import subprocess
import sys

def add_dependencies():
    with open("requirements.txt", "r") as f:
        lines = f.readlines()

    packages = []
    torch_packages = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Strip version for standard packages to avoid "no version found" errors
        # But keep the name.
        if "==" in line:
            pkg_name = line.split("==")[0]
        else:
            pkg_name = line

        if "torch" in pkg_name or "triton" in pkg_name or "nvidia" in pkg_name: 
             # For torch, we might want to keep the version if it's special, 
             # but to be safe and get it working, we'll try latest/compatible first.
             # Actually, for torch, usually compatible is better than unmatched exact.
             torch_packages.append(pkg_name)
        else:
            packages.append(pkg_name)

    # Add standard packages
    if packages:
        print(f"Adding {len(packages)} standard packages (loose versions)...")
        # We process in chunks to avoid command line length issues if any, though 64 is fine.
        subprocess.run(["uv", "add"] + packages, check=False)

    # Add torch packages
        # Add torch packages with standard index resolution (let uv decide)
        # Verify cuda availability post-install if needed
        cmd = ["uv", "add"] + torch_packages
        subprocess.run(cmd, check=False)

if __name__ == "__main__":
    add_dependencies()
