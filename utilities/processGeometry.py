import gmsh
import math
import os
from typing import Set, Union, Tuple, TypeVar, Optional, Dict, Any, Callable
import functools
import importlib
from functools import lru_cache
from dataclasses import dataclass
import numpy as np
import logging
import shutil

logger = logging.getLogger(__name__)

class GeometryProcessor:
    """A class to handle various geometry processing operations."""
    
    SUPPORTED_FORMATS: Set[str] = {'igs', 'iges', 'stp', 'step', 'brep'}

    REQUIRED_PACKAGES = {
        'trimesh': 'trimesh',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'shapely': 'shapely'
    }

    DEFAULT_CONFIG = {
        'mesh_repair': {
            'fill_holes': True,
            'fix_normals': True,
            'merge_vertices': True
        },
        'export': {
            'ascii': False,
            'precision': 10
        }
    }

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with custom configuration"""
        self.config = self._merge_config(self.DEFAULT_CONFIG, config or {})
        self._is_gmsh_initialized = False
        logger.debug("GeometryProcessor initialized")

    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Simple recursive dictionary merge"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _initialize_gmsh(self):
        """Initialize GMSH if not already initialized."""
        if not self._is_gmsh_initialized:
            gmsh.initialize()
            self._is_gmsh_initialized = True

    def _finalize_gmsh(self):
        """Finalize GMSH if it's initialized."""
        if self._is_gmsh_initialized:
            gmsh.finalize()
            self._is_gmsh_initialized = False

    def convert_to_stl(
        self,
        input_file: str,
        output_file: str = None,
        scale: float = 1.0,
        mirror: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Convert CAD files (IGS/STEP/BREP) to STL format.
        
        Args:
            input_file (str): Path to the input CAD file
            output_file (str, optional): Path for the output STL file
            scale (float, optional): Scaling factor. Defaults to 1.0
            mirror (bool, optional): Mirror the geometry in XZ-plane. Defaults to False
            progress_callback: Callable that accepts progress (0-1) and status message
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            logger.info(f"Starting STL conversion of {input_file}")
            if progress_callback:
                progress_callback(0.1, "Loading input file...")
            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            file_extension = input_file.lower().split('.')[-1]
            if file_extension not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported file format: {file_extension}. Supported formats: {self.SUPPORTED_FORMATS}"

            # Set default output file if none provided
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + '.stl'

            # Process the geometry
            self._initialize_gmsh()
            
            # Load the input file
            gmsh.merge(input_file)

            if progress_callback:
                progress_callback(0.5, "Processing geometry...")

            # Join surfaces before classification
            gmsh.model.occ.synchronize()
            # gmsh.model.occ.removeAllDuplicates()
            # gmsh.model.occ.healShapes() # causing problems
            
            # Classify surfaces
            angle = 20
            gmsh.model.mesh.classifySurfaces(
                angle * math.pi / 180.,
                True,  # boundary
                True,  # forceParametrizablePatches
                180 * math.pi / 180.  # curveAngle
            )

            # Recombine into a single surface mesh
            gmsh.model.mesh.recombine()

            # Create volume and set mesh parameters
            gmsh.model.mesh.createGeometry()
            gmsh.option.setNumber("Mesh.Algorithm", 5)
            gmsh.model.mesh.generate(3)

            # Apply scaling
            gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, scale, scale, scale)

            # Apply mirroring if requested
            if mirror:
                entities = gmsh.model.getEntities()
                gmsh.model.occ.copy(entities)
                mirrored_entities = gmsh.model.getEntities()[-len(entities):]
                gmsh.model.occ.mirror(mirrored_entities, 0, 1, 0, 0)
                gmsh.model.occ.fuse(entities, mirrored_entities)

            # Synchronize and repair mesh
            gmsh.model.occ.synchronize()
            
            # # Set repair options
            # gmsh.option.setNumber("Geometry.Tolerance", 1e-8)  # Geometrical tolerance
            # gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)  # Global mesh size factor
            
            # # Additional gap closing options
            # gmsh.option.setNumber("Geometry.AutoCoherence", 1)  # Automatically fix small gaps
            # gmsh.option.setNumber("Geometry.Tolerance", 1e-4)  # Increase tolerance for gap closing
            # gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)  # More aggressive overlap detection
            # gmsh.option.setNumber("Mesh.MinimumCirclePoints", 12)  # Better circular edge discretization
            
            # # Get all surfaces and attempt to fuse them
            # surfaces = gmsh.model.getEntities(2)  # Get all 2D entities (surfaces)
            # if len(surfaces) > 1:
            #     # Attempt to fuse all surfaces into one
            #     gmsh.model.occ.fuse(surfaces[:1], surfaces[1:])
            #     gmsh.model.occ.synchronize()
            
            # # Heal shapes to fix small gaps and inconsistencies
            # # gmsh.model.occ.healShapes()
            # gmsh.model.occ.synchronize()
            
            # # Remove small edges and faces
            # # gmsh.option.setNumber("Geometry.OCCMinimumEdgeLength", 1e-4)  # Minimum edge length
            # # gmsh.option.setNumber("Geometry.OCCMinimumFaceArea", 1e-6)   # Minimum face area
            
            # # Repair operations
            # gmsh.model.mesh.removeDuplicateNodes()  # Remove duplicate nodes
            # gmsh.option.setNumber("Mesh.ScalingFactor", 1.0)  # Reset scaling
            # gmsh.model.mesh.createTopology()  # Create topology from mesh
            # gmsh.model.mesh.classifySurfaces(math.pi)  # Classify surfaces with angle tolerance
            # gmsh.model.mesh.createGeometry()  # Create geometry from topology
            
            # # Additional repair options for gaps
            # gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 1)  # Create one solid per surface
            # gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)  # Remove duplicate triangles
            # gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay for surface meshing
            # gmsh.option.setNumber("Mesh.Binary", 0)  # ASCII output for better precision
            
            # # Generate and optimize mesh with additional steps
            # gmsh.model.mesh.generate(2)  # Generate 2D mesh
            
            # # More aggressive mesh optimization
            # gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)  # Lower threshold for optimization
            # gmsh.model.mesh.optimize("Netgen", niter=10)  # More iterations of Netgen optimization
            # gmsh.model.mesh.optimize("Laplace2D", niter=10)  # More iterations of Laplace smoothing
            # gmsh.model.mesh.optimize("Relocate2D")  # Relocate vertices for better quality
            
            # # Final coherence check and repair
            # gmsh.model.occ.removeAllDuplicates()
            # gmsh.model.occ.synchronize()
            
            # Write the repaired mesh
            gmsh.write(output_file)
          
            # Create a copy named hull_convert.stl
            hull_convert_path = os.path.join(os.path.dirname(output_file), 'hull_convert.stl')
            shutil.copy2(output_file, hull_convert_path)            
            logger.info(f"Created copy at: {hull_convert_path}")

            logger.debug(f"Applied transformation: scale={scale}, mirror={mirror}")

            # Load the STL file using trimesh to get bounding box and dimensions
            import trimesh
            mesh = trimesh.load_mesh(output_file)
            bbox = mesh.bounds
            dimensions = mesh.extents

            logger.info(f"Bounding box: {bbox}")
            logger.info(f"Model dimensions: {dimensions}")

            return True, (
                f"Successfully converted {input_file} to {output_file}. "
                f"Bounding box: min [{bbox[0][0]:.2f}, {bbox[0][1]:.2f}, {bbox[0][2]:.2f}], "
                f"max [{bbox[1][0]:.2f}, {bbox[1][1]:.2f}, {bbox[1][2]:.2f}]. "
                f"Model dimensions: [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]"
            )

        except Exception as e:
            logger.error(f"STL conversion failed: {str(e)}", exc_info=True)
            return False, f"Error during conversion: {str(e)}"

        finally:
            self._finalize_gmsh()

    def close_openings(
        self,
        input_file: str,
        output_file: str = None,
        method: str = 'trimesh'
    ) -> Tuple[bool, str]:
        """
        Close openings in an STL mesh by extruding open edges to the XZ plane.
        
        Args:
            input_file (str): Path to the input STL file
            output_file (str, optional): Path for the output STL file
            method (str, optional): Method to use ('trimesh' or 'gmsh'). Defaults to 'trimesh'
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        if method == 'trimesh':
            return self._close_openings_trimesh(input_file, output_file)
        elif method == 'gmsh':
            return self._close_openings_gmsh(input_file, output_file)
        else:
            return False, f"Unsupported method: {method}. Use 'trimesh' or 'gmsh'."

    def _close_openings_trimesh(self, input_file: str, output_file: str = None) -> Tuple[bool, str]:
        """Internal method using trimesh to close openings."""
        try:
            import trimesh
            import numpy as np

            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            # Set default output file if none provided
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + '_closed.stl'

            # Load the mesh
            mesh = trimesh.load_mesh(input_file)
            initial_state = {
                "is_watertight": mesh.is_watertight,
                "volume": mesh.volume,
                "area": mesh.area,
                "center_mass": mesh.center_mass,
                "extents": mesh.extents,
                "bounds": mesh.bounds
            }

            if mesh.is_watertight:
                return True, "Mesh is already watertight. No modifications needed."

            # Process the mesh
            modified_mesh = self._extrude_edges_to_xz(mesh)
            
            if not modified_mesh.is_watertight:
                # Try additional repairs
                trimesh.repair.fill_holes(modified_mesh)
                trimesh.repair.fix_normals(modified_mesh)
                trimesh.repair.fix_inversion(modified_mesh)
                trimesh.repair.fix_winding(modified_mesh)
                trimesh.repair.broken_faces(modified_mesh)
                modified_mesh.remove_duplicate_faces()
                modified_mesh.remove_degenerate_faces()
                modified_mesh.remove_infinite_values()
                modified_mesh.merge_vertices(merge_tex=True)
                modified_mesh.process()

            # Export the result
            modified_mesh.export(output_file)

            final_state = {
                "is_watertight": modified_mesh.is_watertight,
                "volume": modified_mesh.volume,
                "area": modified_mesh.area,
                "center_mass": modified_mesh.center_mass,
                "extents": modified_mesh.extents,
                "bounds": modified_mesh.bounds
            }

            status_message = (
                f"Processing complete:\n"
                f"Initial watertight: {initial_state['is_watertight']} → Final: {final_state['is_watertight']}\n"
                f"Volume: {self._format_number(initial_state['volume'])} → {self._format_number(final_state['volume'])}\n"
                f"Surface area: {self._format_number(initial_state['area'])} → {self._format_number(final_state['area'])}"
            )

            if not modified_mesh.is_watertight:
                return False, f"{self.YELLOW}Warning: Mesh is still not watertight. {status_message}{self.RESET}"
            
            return True, f"{self.GREEN}Successfully created watertight mesh. {status_message}{self.RESET}"

        except Exception as e:
            return False, f"Error during mesh processing: {str(e)}"

    def _close_openings_gmsh(self, input_file: str, output_file: str = None) -> Tuple[bool, str]:
        """Internal method using GMSH to close openings."""
        try:
            import numpy as np

            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            # Set default output file if none provided
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + '_closed.stl'

            self._initialize_gmsh()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("close_openings")

            # Merge the STL file
            gmsh.merge(input_file)

            # Get mesh data
            nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

            # Build node mapping
            tag_to_local = {}
            coords_array = []
            for i, tag in enumerate(nodeTags):
                tag_to_local[tag] = i
                coords_array.append((
                    nodeCoords[3*i],
                    nodeCoords[3*i + 1],
                    nodeCoords[3*i + 2]
                ))
            coords_array = np.array(coords_array)

            # Get triangles
            eTypes, eIds, eNodes = gmsh.model.mesh.getElements(dim=2)
            all_triangles = []
            for etype, ids, nodes in zip(eTypes, eIds, eNodes):
                if etype != 2:  # type 2 = triangle
                    continue
                for i in range(len(ids)):
                    base = 3 * i
                    all_triangles.append((
                        nodes[base],
                        nodes[base + 1],
                        nodes[base + 2]
                    ))
            all_triangles = np.array(all_triangles, dtype=int)

            # Find boundary edges
            edge_dict = {}
            for tri in all_triangles:
                edges = [
                    tuple(sorted((tri[0], tri[1]))),
                    tuple(sorted((tri[1], tri[2]))),
                    tuple(sorted((tri[2], tri[0]))),
                ]
                for e in edges:
                    edge_dict[e] = edge_dict.get(e, 0) + 1

            boundary_edges = [e for e, count in edge_dict.items() if count == 1]
            
            if not boundary_edges:
                return True, "Mesh is already watertight. No modifications needed."

            # Create new nodes and triangles
            new_node_coords = []
            new_node_tags = []
            new_triangles = []
            next_tag = max(nodeTags) + 1
            projected_node_map = {}

            # Project and create new nodes/triangles
            for edge in boundary_edges:
                v1, v2 = edge
                
                # Get or create projected nodes
                for v in (v1, v2):
                    if v not in projected_node_map:
                        old_coords = coords_array[tag_to_local[v]]
                        projected_node_map[v] = next_tag
                        new_node_tags.append(next_tag)
                        new_node_coords.append((old_coords[0], 0.0, old_coords[2]))
                        next_tag += 1

                pv1, pv2 = projected_node_map[v1], projected_node_map[v2]
                new_triangles.append([v1, v2, pv1])
                new_triangles.append([pv1, v2, pv2])

            # Add new nodes to the model
            if new_node_coords:
                gmsh.model.mesh.addNodes(
                    dim=2,
                    tag=1,
                    nodeTags=new_node_tags,
                    coord=[c for xyz in new_node_coords for c in xyz]
                )

                # Add new triangles
                _, existing_elem_ids, _ = gmsh.model.mesh.getElements(dim=2)
                flattened_ids = np.concatenate(existing_elem_ids) if existing_elem_ids else []
                next_elem_id = max(flattened_ids) + 1 if len(flattened_ids) > 0 else 1
                
                new_elem_tags = list(range(next_elem_id, next_elem_id + len(new_triangles)))
                new_elem_node_tags = [node for tri in new_triangles for node in tri]

                gmsh.model.mesh.addElements(
                    dim=2,
                    tag=1,
                    elementTypes=[2],
                    elementTags=[new_elem_tags],
                    nodeTags=[new_elem_node_tags]
                )

            # Write the result
            gmsh.write(output_file)
            
            return True, f"Successfully closed {len(boundary_edges)} open edges using GMSH method"

        except Exception as e:
            return False, f"Error during GMSH processing: {str(e)}"
        
        finally:
            self._finalize_gmsh()

    def _extrude_edges_to_xz(self, mesh: 'trimesh.Trimesh') -> 'trimesh.Trimesh':
        """
        Helper method to extrude open edges to the XZ plane.
        
        Args:
            mesh: Input trimesh mesh
            
        Returns:
            Modified trimesh mesh
        """
        import trimesh
        import numpy as np

        # Find open edges
        edges = mesh.edges
        edges_sorted = np.sort(edges, axis=1)
        edge_counts = trimesh.grouping.group_rows(edges_sorted, require_count=1)
        open_edges = edges[edge_counts]

        if len(open_edges) == 0:
            return mesh

        # Collect new vertices and faces
        new_vertices = []
        new_faces = []

        for edge in open_edges:
            v1, v2 = mesh.vertices[edge]

            # Create new vertices projected onto the XZ plane
            v1_proj = [v1[0], 0, v1[2]]
            v2_proj = [v2[0], 0, v2[2]]

            # Add new vertices and faces
            base_index = len(mesh.vertices) + len(new_vertices)
            new_vertices.extend([v1_proj, v2_proj])
            new_faces.extend([
                [edge[0], edge[1], base_index],
                [base_index, base_index + 1, edge[1]]
            ])

        # Combine everything into a new mesh
        new_vertices = np.array(new_vertices)
        new_faces = np.array(new_faces)

        return trimesh.Trimesh(
            vertices=np.vstack([mesh.vertices, new_vertices]),
            faces=np.vstack([mesh.faces, new_faces]),
            process=False
        )

    def mirror_geometry(
        self,
        input_file: str,
        output_file: str = None,
        mirror_axis: str = 'y',
        origin: bool = True
    ) -> Tuple[bool, str]:
        """
        Mirror a mesh along a specified axis.
        
        Args:
            input_file (str): Path to the input mesh file
            output_file (str, optional): Path for the output mesh file
            mirror_axis (str, optional): Axis to mirror along ('x', 'y', or 'z'). Defaults to 'y'
            origin (bool, optional): If True, mirror through origin. If False, mirror through mesh center. Defaults to True
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            import trimesh
            import numpy as np

            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            # Set default output file if none provided
            if output_file is None:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_mirrored{ext}"

            # Validate mirror axis
            mirror_axis = mirror_axis.lower()
            if mirror_axis not in ['x', 'y', 'z']:
                return False, "Mirror axis must be 'x', 'y', or 'z'"

            # Load the mesh
            mesh = trimesh.load_mesh(input_file)
            initial_state = {
                "volume": mesh.volume,
                "area": mesh.area,
                "center_mass": mesh.center_mass,
                "bounds": mesh.bounds
            }

            # Create transformation matrix for mirroring
            mirror_matrix = np.eye(4)
            axis_index = {'x': 0, 'y': 1, 'z': 2}[mirror_axis]
            mirror_matrix[axis_index, axis_index] = -1

            if not origin:
                # Mirror through mesh center instead of origin
                center = mesh.center_mass
                translation = np.eye(4)
                translation[:3, 3] = center
                inverse_translation = np.eye(4)
                inverse_translation[:3, 3] = -center
                mirror_matrix = translation @ mirror_matrix @ inverse_translation

            # Create mirrored copy
            mirrored_mesh = mesh.copy()
            mirrored_mesh.apply_transform(mirror_matrix)

            # Combine original and mirrored meshes
            combined_vertices = np.vstack((mesh.vertices, mirrored_mesh.vertices))
            combined_faces = np.vstack((
                mesh.faces,
                mirrored_mesh.faces + len(mesh.vertices)
            ))

            combined_mesh = trimesh.Trimesh(
                vertices=combined_vertices,
                faces=combined_faces,
                process=True
            )

            # Export the result
            combined_mesh.export(output_file)

            final_state = {
                "volume": combined_mesh.volume,
                "area": combined_mesh.area,
                "center_mass": combined_mesh.center_mass,
                "bounds": combined_mesh.bounds
            }

            status_message = (
                f"Mirroring complete:\n"
                f"Volume: {self._format_number(initial_state['volume'])} → {self._format_number(final_state['volume'])}\n"
                f"Surface area: {self._format_number(initial_state['area'])} → {self._format_number(final_state['area'])}\n"
                f"Center of mass: {self._format_vector(final_state['center_mass'])}\n"
                f"Bounds: min {self._format_vector(final_state['bounds'][0])}, "
                f"max {self._format_vector(final_state['bounds'][1])}"
            )

            return True, f"{self.GREEN}Successfully mirrored mesh along {mirror_axis}-axis. {status_message}{self.RESET}"

        except Exception as e:
            return False, f"Error during mirroring: {str(e)}"

    def transform_geometry(
        self,
        input_file: str,
        output_file: str = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        translate: Tuple[float, float, float] = (0, 0, 0),
        rotate: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[bool, str]:
        """
        Apply geometric transformations (scale, translate, rotate) to a mesh.
        
        Args:
            input_file (str): Path to the input mesh file
            output_file (str, optional): Path for the output mesh file
            scale (float or tuple, optional): Scale factor. If float, applies uniform scaling.
                If tuple, applies (x, y, z) scaling. Defaults to 1.0
            translate (tuple, optional): Translation vector (x, y, z). Defaults to (0, 0, 0)
            rotate (tuple, optional): Rotation angles in degrees (rx, ry, rz) applied in order:
                first around x, then y, then z axis. Defaults to (0, 0, 0)
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            import trimesh
            import numpy as np

            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            # Set default output file if none provided
            if output_file is None:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_transformed{ext}"

            # Load the mesh
            mesh = trimesh.load_mesh(input_file)
            initial_state = {
                "is_watertight": mesh.is_watertight,
                "volume": mesh.volume,
                "area": mesh.area,
                "center_mass": mesh.center_mass,
                "bounds": mesh.bounds,
                "dimensions": mesh.extents
            }

            # Handle scaling
            if isinstance(scale, (int, float)):
                scale = (float(scale), float(scale), float(scale))
            scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
            mesh.apply_transform(scale_matrix)

            # Handle rotation (convert degrees to radians)
            for axis, angle in enumerate(rotate):
                if angle != 0:
                    rad_angle = np.radians(angle)
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        rad_angle, 
                        [1.0 if axis == 0 else 0.0, 1.0 if axis == 1 else 0.0, 1.0 if axis == 2 else 0.0]
                    )
                    mesh.apply_transform(rotation_matrix)

            # Handle translation
            translation_matrix = trimesh.transformations.translation_matrix(translate)
            mesh.apply_transform(translation_matrix)

            # Export the result
            mesh.export(output_file)

            final_state = {
                "is_watertight": mesh.is_watertight,
                "volume": mesh.volume,
                "area": mesh.area,
                "center_mass": mesh.center_mass,
                "bounds": mesh.bounds,
                "dimensions": mesh.extents
            }

            status_message = (
                f"Processing complete:\n"
                f"Initial watertight: {initial_state['is_watertight']} → Final: {final_state['is_watertight']}\n"
                f"Volume: {self._format_number(initial_state['volume'])} → {self._format_number(final_state['volume'])}\n"
                f"Surface area: {self._format_number(initial_state['area'])} → {self._format_number(final_state['area'])}\n"
                f"Initial bounding box:\n"
                f"  Min: {self._format_vector(initial_state['bounds'][0])}\n"
                f"  Max: {self._format_vector(initial_state['bounds'][1])}\n"
                f"Final bounding box:\n"
                f"  Min: {self._format_vector(final_state['bounds'][0])}\n"
                f"  Max: {self._format_vector(final_state['bounds'][1])}\n"
                f"Initial dimensions (x,y,z): {self._format_vector(initial_state['dimensions'])}\n"
                f"Final dimensions (x,y,z): {self._format_vector(final_state['dimensions'])}"
            )

            return True, f"{self.GREEN}Successfully transformed mesh. {status_message}{self.RESET}"

        except Exception as e:
            return False, f"Error during transformation: {str(e)}"

    def cut_by_draft(
        self,
        input_file: str,
        draft: float,
        output_file: str = None,
        close_method: str = 'trimesh'
    ) -> Tuple[bool, str]:
        """
        Cut geometry by z-plane at specified draft height, keep lower part and close the opening.
        
        Args:
            input_file (str): Path to the input mesh file
            draft (float): Height of the cutting plane in z-axis
            output_file (str, optional): Path for the output mesh file
            close_method (str, optional): Method to use for closing ('trimesh' or 'gmsh'). Defaults to 'trimesh'
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            # Check for required dependencies
            try:
                import trimesh
                import scipy
                import shapely
            except ImportError as e:
                missing_package = str(e).split("'")[1]
                return False, f"{missing_package} is required for cutting operations. Please install it using 'pip install {missing_package}'"

            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            # Set default output file if none provided
            if output_file is None:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_draft{ext}"

            # Load the mesh
            mesh = trimesh.load_mesh(input_file)
            initial_state = {
                "volume": mesh.volume,
                "area": mesh.area,
                "bounds": mesh.bounds,
                "centroid": mesh.centroid
            }

            # Create cutting plane
            # Normal pointing up to keep lower part
            plane_normal = [0, 0, -1]
            plane_origin = [0, 0, draft]

            # Slice the mesh
            cut_mesh = trimesh.intersections.slice_mesh_plane(
                mesh=mesh,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                cap=False
            )

            if cut_mesh is None:
                return False, "Failed to cut mesh - possibly no intersection with cutting plane"

            # Close the opening
            temp_file = os.path.splitext(output_file)[0] + '_temp.stl'
            cut_mesh.export(temp_file)
            
            success, message = self.close_openings(temp_file, output_file, method=close_method)
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # if not success:
            #     return False, f"Failed to close openings: {message}"

            # Load final mesh for measurements
            final_mesh = trimesh.load_mesh(output_file)
            final_state = {
                "volume": final_mesh.volume,
                "area": final_mesh.area,
                "bounds": final_mesh.bounds,
                "centroid": final_mesh.centroid
            }

            status_message = (
                f"Processing complete:\n"
                f"Cut at z = {draft}\n"
                f"Volume: {self._format_number(initial_state['volume'])} → {self._format_number(final_state['volume'])}\n"
                f"Surface area: {self._format_number(initial_state['area'])} → {self._format_number(final_state['area'])}\n"
                f"Initial bounding box:\n"
                f"  Min: {self._format_vector(initial_state['bounds'][0])}\n"
                f"  Max: {self._format_vector(initial_state['bounds'][1])}\n"
                f"Final bounding box:\n"
                f"  Min: {self._format_vector(final_state['bounds'][0])}\n"
                f"  Max: {self._format_vector(final_state['bounds'][1])}\n"
                f"Centroid:\n"
                f"  Initial: {self._format_vector(initial_state['centroid'])}\n"
                f"  Final: {self._format_vector(final_state['centroid'])}"
            )

            return True, f"Successfully cut and closed mesh. {status_message}"

        except Exception as e:
            return False, f"Error during cutting: {str(e)}"

    def write_hull_bounds(
        self,
        input_file: str,
        draft: float,
        output_file: str = 'hullBounds.txt'
    ) -> Tuple[bool, str]:
        """
        Write bounding box values from an STL file to a text file.
        
        Args:
            input_file (str): Path to the input STL file
            draft (float): Draft height (waterline)
            output_file (str, optional): Path for the output text file. Defaults to 'hullBounds.txt'
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            import trimesh

            # Load the mesh
            mesh = trimesh.load_mesh(input_file)
            if mesh is None:
                return False, f"Failed to load mesh from {input_file}"

            # Get bounding box corners
            bbox = mesh.bounds
            # bbox is [[xmin, ymin, zmin], [xmax, ymax, zmax]]

            with open(output_file, 'w') as f:
                f.write(f"hullXmin  {bbox[0][0]:.4f};\n")
                f.write(f"hullXmax  {bbox[1][0]:.4f};\n")
                f.write(f"hullYmin  {bbox[0][1]:.4f};\n")
                # Assuming the hull is symmetric about Y-axis
                f.write(f"hullYmax  {0.0};\n")
                f.write(f"hullZmin  {bbox[0][2]:.4f};\n")
                f.write(f"hullZmax  {bbox[1][2]:.4f};\n")
                f.write(f"zWL       {draft:.4f};\n")

            return True, f"Successfully wrote bounding box to {output_file}"

        except Exception as e:
            return False, f"Error writing bounding box: {str(e)}"

    def approximate_mass_properties(
        self,
        original_stl: str,
        clipped_stl: str,
        rho_water: float = 1000.0,
        output_file: str = 'hullMassInertiaCoG.txt'
    ) -> Tuple[bool, str]:
        """
        Calculate approximate mass properties from original and clipped hull geometries.
        
        Args:
            original_stl (str): Path to the original hull STL file
            clipped_stl (str): Path to the clipped (underwater portion) STL file
            rho_water (float, optional): Water density in kg/m^3. Defaults to 1000.0
            output_file (str, optional): Output file path. Defaults to 'hullMassInertiaCoG.txt'
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            import trimesh
            import numpy as np

            # Load both meshes
            original_mesh = trimesh.load_mesh(original_stl)
            clipped_mesh = trimesh.load_mesh(clipped_stl)
            
            if original_mesh is None or clipped_mesh is None:
                return False, "Failed to load one or both mesh files"

            # Get dimensions from original hull
            dim = original_mesh.extents  # [x_extent, y_extent, z_extent]
            Length = dim[0] * 0.94  # Length LPP approximation
            Beam = dim[1]           # Beam of the hull
            Depth = dim[2]          # Depth of the hull
            VCG = Depth * 0.65      # Approximate Vertical Center of Gravity

            # Get volume from clipped (underwater) portion
            vol = abs(clipped_mesh.volume)  # Ensure volume is positive

            # Calculate mass (assuming half hull)
            mass = vol * rho_water / 2

            # Calculate radii of gyration
            kxx = 0.34 * Beam
            kyy = 0.25 * Length
            kzz = 0.26 * Length

            # Calculate moments of inertia
            Ixx = mass * kxx**2
            Iyy = mass * kyy**2
            Izz = mass * kzz**2

            # Get center of buoyancy (centroid of underwater volume)
            CoB = clipped_mesh.center_mass
            
            # Approximate CoG (using CoB x,y and VCG for z)
            CoG = (CoB[0], CoB[1], VCG)

            # Write results to file
            with open(output_file, "w") as f:
                f.write(f"mass            {mass:.2f};\n")
                f.write(f"Ixx             {Ixx:.2f};\n")
                f.write(f"Iyy             {Iyy:.2f};\n")
                f.write(f"Izz             {Izz:.2f};\n")
                f.write(f"centreOfMass    ({CoG[0]:.6f} {CoG[1]:.6f} {CoG[2]:.6f});\n")

            status_message = (
                f"Mass properties calculated:\n"
                f"Hull dimensions (L x B x D): {self._format_vector([Length, Beam, Depth])}\n"
                f"Displacement: {self._format_number(mass)} kg\n"
                f"CoG: {self._format_vector(CoG)}\n"
                f"Results written to {output_file}"
            )

            return True, f"{self.GREEN}{status_message}{self.RESET}"

        except Exception as e:
            return False, f"Error calculating mass properties: {str(e)}"

    def remesh_geometry(
        self,
        input_file: str,
        output_file: str = None,
        max_hole_size: float = 1000.0,
        target_edge_length: float = 1.0,
        iterations: int = 5,
        min_component_size: int = 100
    ) -> Tuple[bool, str]:
        """
        Remesh and optimize the geometry using PyMeshLab.
        
        Args:
            input_file (str): Path to the input mesh file
            output_file (str, optional): Path for the output mesh file
            max_hole_size (float, optional): Maximum area of holes to fill. Defaults to 1000.0
            target_edge_length (float, optional): Desired average edge length. Defaults to 1.0
            iterations (int, optional): Number of remeshing iterations. Defaults to 5
            min_component_size (int, optional): Minimum number of faces for components. Defaults to 100
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
            import pymeshlab
            import trimesh

            # Validate input file
            if not os.path.exists(input_file):
                return False, f"Input file not found: {input_file}"

            # Set default output file if none provided
            if output_file is None:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_remeshed{ext}"

            # Load initial mesh state for comparison
            initial_mesh = trimesh.load_mesh(input_file)
            initial_state = {
                "vertices": len(initial_mesh.vertices),
                "faces": len(initial_mesh.faces),
                "volume": initial_mesh.volume,
                "area": initial_mesh.area,
                "is_watertight": initial_mesh.is_watertight
            }

            # Create a MeshSet and load the mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(input_file)

            # Remove duplicate faces and vertices
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()

            # Fill/close small holes
            try:
                ms.meshing_close_holes(maxholesize=int(max_hole_size))
            except:
                # If integer conversion fails, try without maxholesize parameter
                ms.meshing_close_holes()

            # Get current mesh bounding box to calculate appropriate target length
            current_mesh = ms.current_mesh()
            bbox = current_mesh.bounding_box()
            bbox_diag = ((bbox.max()[0] - bbox.min()[0])**2 + 
                        (bbox.max()[1] - bbox.min()[1])**2 + 
                        (bbox.max()[2] - bbox.min()[2])**2)**0.5
            relative_length = target_edge_length / bbox_diag * 100  # Convert to percentage

            # Perform isotropic explicit remeshing
            try:
                ms.meshing_isotropic_explicit_remeshing(
                    iterations=int(iterations),
                    targetlen=relative_length
                )
            except:
                # Fallback to default parameters if explicit parameters fail
                ms.meshing_isotropic_explicit_remeshing()

            # Remove small disconnected components
            try:
                ms.meshing_remove_connected_component_by_face_number(
                    mincomponentsize=int(min_component_size)
                )
            except:
                # Fallback to default parameters if explicit parameters fail
                ms.meshing_remove_connected_component_by_face_number()

            # Final cleanup
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()

            # Save the result
            ms.save_current_mesh(output_file)
            
            # Create a copy named hull_remeshed.stl
            hull_remeshed_path = os.path.join(os.path.dirname(output_file), 'hull_remeshed.stl')
            shutil.copy2(output_file, hull_remeshed_path)
            logger.info(f"Created copy at: {hull_remeshed_path}")

            # Load final mesh state for comparison
            final_mesh = trimesh.load_mesh(output_file)
            final_state = {
                "vertices": len(final_mesh.vertices),
                "faces": len(final_mesh.faces),
                "volume": final_mesh.volume,
                "area": final_mesh.area,
                "is_watertight": final_mesh.is_watertight
            }

            status_message = (
                f"Remeshing complete:\n"
                f"Vertices: {self._format_number(initial_state['vertices'])} → "
                f"{self._format_number(final_state['vertices'])}\n"
                f"Faces: {self._format_number(initial_state['faces'])} → "
                f"{self._format_number(final_state['faces'])}\n"
                f"Volume: {self._format_number(initial_state['volume'])} → "
                f"{self._format_number(final_state['volume'])}\n"
                f"Surface area: {self._format_number(initial_state['area'])} → "
                f"{self._format_number(final_state['area'])}\n"
                f"Watertight: {initial_state['is_watertight']} → {final_state['is_watertight']}"
            )

            return True, f"{self.GREEN}Successfully remeshed geometry. {status_message}{self.RESET}"

        except ImportError:
            return False, "PyMeshLab is required for remeshing. Install with: pip install pymeshlab"
        except Exception as e:
            return False, f"Error during remeshing: {str(e)}"

    def __enter__(self):
        """Enable context manager support for automatic GMSH cleanup"""
        self._initialize_gmsh()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup GMSH when exiting context"""
        self._finalize_gmsh()

    @lru_cache(maxsize=32)
    def _load_mesh(self, file_path: str) -> 'trimesh.Trimesh':
        """Cache mesh loading to avoid repeated disk reads"""
        import trimesh
        return trimesh.load_mesh(file_path)

    def _format_number(self, num: float) -> str:
        """Format number with cyan color"""
        return f"{self.CYAN}{num:.2f}{self.RESET}"

    def _format_vector(self, vec) -> str:
        """Format vector of numbers with cyan color"""
        return f"[{', '.join(self._format_number(x) for x in vec)}]"

def check_dependencies(*packages):
    """Decorator to check required packages before executing a method"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing = []
            for pkg in packages:
                try:
                    importlib.import_module(pkg)
                except ImportError:
                    missing.append(pkg)
            if missing:
                return False, f"Missing required packages: {', '.join(missing)}. Install with pip install {' '.join(missing)}"
            return func(*args, **kwargs)
        return wrapper
    return decorator

T = TypeVar('T', bound='GeometryProcessor')

@dataclass
class MeshState:
    """Data class to hold mesh state information"""
    is_watertight: bool
    volume: float
    area: float
    center_mass: np.ndarray
    bounds: np.ndarray

def get_mesh_state(self, mesh: 'trimesh.Trimesh') -> MeshState:
    """Get current state of mesh properties"""
    return MeshState(
        is_watertight=mesh.is_watertight,
        volume=mesh.volume,
        area=mesh.area,
        center_mass=mesh.center_mass,
        bounds=mesh.bounds
    )
