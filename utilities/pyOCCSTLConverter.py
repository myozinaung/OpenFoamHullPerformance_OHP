from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRepTools import breptools_Read
import os

class ModelToStlConverter:
    """
    A class to convert various 3D model formats (STEP, IGES, BREP) to STL format
    using pythonOCC.
    """
    
    @staticmethod
    def convert_step_to_stl(input_file: str, output_file: str, linear_deflection: float = 0.1):
        """Convert STEP file to STL"""
        # Initialize the STEP reader
        step_reader = STEPControl_Reader()
        
        # Read the STEP file
        status = step_reader.ReadFile(input_file)
        if status != IFSelect_RetDone:
            raise Exception(f"Error reading STEP file: {input_file}")
            
        # Transfer shapes and check
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        
        # Convert to STL
        ModelToStlConverter._shape_to_stl(shape, output_file, linear_deflection)
        
    @staticmethod
    def convert_iges_to_stl(input_file: str, output_file: str, linear_deflection: float = 0.1):
        """Convert IGES file to STL"""
        # Initialize the IGES reader
        iges_reader = IGESControl_Reader()
        
        # Read the IGES file
        status = iges_reader.ReadFile(input_file)
        if status != IFSelect_RetDone:
            raise Exception(f"Error reading IGES file: {input_file}")
            
        # Transfer shapes and check
        iges_reader.TransferRoots()
        shape = iges_reader.OneShape()
        
        # Convert to STL
        ModelToStlConverter._shape_to_stl(shape, output_file, linear_deflection)
        
    @staticmethod
    def convert_brep_to_stl(input_file: str, output_file: str, linear_deflection: float = 0.1):
        """Convert BREP file to STL"""
        # Initialize shape and builder
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        
        # Read the BREP file
        result = breptools_Read(shape, input_file, builder)
        if not result:
            raise Exception(f"Error reading BREP file: {input_file}")
            
        # Convert to STL
        ModelToStlConverter._shape_to_stl(shape, output_file, linear_deflection)
    
    @staticmethod
    def _shape_to_stl(shape: TopoDS_Shape, output_file: str, linear_deflection: float):
        """Helper method to convert a TopoDS_Shape to STL"""
        # Create a mesh from the shape
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection)
        mesh.Perform()
        
        # Write to STL
        stl_writer = StlAPI_Writer()
        stl_writer.Write(shape, output_file)

def convert_to_stl(input_file: str, output_file: str = None, linear_deflection: float = 0.1):
    """
    Main function to convert supported 3D model files to STL format.
    
    Args:
        input_file (str): Path to input file (STEP, IGES, or BREP)
        output_file (str, optional): Path to output STL file. If None, uses input filename with .stl extension
        linear_deflection (float, optional): Mesh quality parameter. Lower values create finer meshes
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # If output_file is not specified, create one based on input filename
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".stl"
    
    # Get file extension and convert to lowercase
    file_ext = os.path.splitext(input_file)[1].lower()
    
    converter = ModelToStlConverter()
    
    try:
        if file_ext == '.step' or file_ext == '.stp':
            converter.convert_step_to_stl(input_file, output_file, linear_deflection)
        elif file_ext == '.iges' or file_ext == '.igs':
            converter.convert_iges_to_stl(input_file, output_file, linear_deflection)
        elif file_ext == '.brep' or file_ext == '.brp':
            converter.convert_brep_to_stl(input_file, output_file, linear_deflection)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "path/to/your/model.step"
    convert_to_stl(input_file)


# # Basic usage
# convert_to_stl("path/to/your/model.step")

# # With custom output path and mesh quality
# convert_to_stl("input.step", "output.stl", linear_deflection=0.05)