import gmsh
import math
import os
from typing import Set, Union, Tuple

class GeometryProcessor:
    """A class to handle various geometry processing operations."""
    
    SUPPORTED_FORMATS: Set[str] = {'igs', 'iges', 'stp', 'step', 'brep'}

    def __init__(self):
        """Initialize the GeometryProcessor."""
        self._is_gmsh_initialized = False

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
        mirror: bool = False
    ) -> Tuple[bool, str]:
        """
        Convert CAD files (IGS/STEP/BREP) to STL format.
        
        Args:
            input_file (str): Path to the input CAD file
            output_file (str, optional): Path for the output STL file
            scale (float, optional): Scaling factor. Defaults to 1.0
            mirror (bool, optional): Mirror the geometry in XZ-plane. Defaults to False
            
        Returns:
            Tuple[bool, str]: (Success status, Message/Error description)
        """
        try:
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

            # Classify surfaces
            angle = 20
            gmsh.model.mesh.classifySurfaces(
                angle * math.pi / 180.,
                True,  # boundary
                True,  # forceParametrizablePatches
                180 * math.pi / 180.  # curveAngle
            )

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

            # Synchronize and save
            gmsh.model.occ.synchronize()
            gmsh.write(output_file)

            return True, f"Successfully converted {input_file} to {output_file}"

        except Exception as e:
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
                f"Volume: {initial_state['volume']:.2f} → {final_state['volume']:.2f}\n"
                f"Surface area: {initial_state['area']:.2f} → {final_state['area']:.2f}"
            )

            if not modified_mesh.is_watertight:
                return False, f"Warning: Mesh is still not watertight. {status_message}"
            
            return True, f"Successfully created watertight mesh. {status_message}"

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
                f"Volume: {initial_state['volume']:.2f} → {final_state['volume']:.2f}\n"
                f"Surface area: {initial_state['area']:.2f} → {final_state['area']:.2f}\n"
                f"Center of mass: [{', '.join(f'{x:.2f}' for x in final_state['center_mass'])}]\n"
                f"Bounds: min [{', '.join(f'{x:.2f}' for x in final_state['bounds'][0])}], "
                f"max [{', '.join(f'{x:.2f}' for x in final_state['bounds'][1])}]"
            )

            return True, f"Successfully mirrored mesh along {mirror_axis}-axis. {status_message}"

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
                "bounds": mesh.bounds
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
                "bounds": mesh.bounds
            }

            status_message = (
                f"Processing complete:\n"
                f"Initial watertight: {initial_state['is_watertight']} → Final: {final_state['is_watertight']}\n"
                f"Volume: {initial_state['volume']:.2f} → {final_state['volume']:.2f}\n"
                f"Surface area: {initial_state['area']:.2f} → {final_state['area']:.2f}"
            )

            return True, f"Successfully transformed mesh. {status_message}"

        except Exception as e:
            return False, f"Error during transformation: {str(e)}"

    def __del__(self):
        """Ensure GMSH is properly finalized when the object is destroyed."""
        self._finalize_gmsh()
