#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <vtkCell.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include "bat_file.h"
#include "lba_tree_builder.h"

const std::string USAGE = "Usage: ./vtu_particle_converter <file.vtu> <out.bat>";

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cout << USAGE << "\n";
        return 1;
    }
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h") {
            std::cout << USAGE << "\n";
            return 0;
        }
    }

    std::cout << "Converting " << args[1] << "\n";

    auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(argv[1]);
    reader->Update();
    auto *mesh = reader->GetOutput();

    mesh->Print(std::cout);

    // Note: this assumes scivis 2016 contest data for this filtering by concentration
    auto *point_data = mesh->GetPointData();
    point_data->Print(std::cout);
    std::cout << "# of arrays: " << point_data->GetNumberOfArrays() << "\n";

    std::vector<Attribute> attributes;
    for (size_t i = 0; i < point_data->GetNumberOfArrays(); ++i) {
        DTYPE type = UNKNOWN;
        vtkDataArray *arr = point_data->GetArray(i);
        arr->Print(std::cout);
        const int vtk_type = arr->GetDataType();
        switch (vtk_type) {
        case VTK_CHAR:
        case VTK_UNSIGNED_CHAR:
            type = UINT_8;
            break;
        case VTK_SIGNED_CHAR:
            type = INT_8;
            break;
        case VTK_SHORT:
            type = INT_16;
            break;
        case VTK_UNSIGNED_SHORT:
            type = UINT_16;
            break;
        case VTK_INT:
            type = INT_32;
            break;
        case VTK_UNSIGNED_INT:
            type = UINT_32;
            break;
        case VTK_LONG:
            type = INT_64;
            break;
        case VTK_UNSIGNED_LONG:
            type = UINT_64;
            break;
        case VTK_FLOAT:
            type = FLOAT_32;
            break;
        case VTK_DOUBLE:
            type = FLOAT_64;
            break;
        default:
            break;
        }

        auto data = std::make_shared<OwnedArray<uint8_t>>();
        for (size_t j = 0; j < arr->GetNumberOfValues(); ++j) {
            uint8_t *x = reinterpret_cast<uint8_t *>(arr->GetVoidPointer(j));
            for (size_t k = 0; k < dtype_stride(type); ++k) {
                data->push_back(x[k]);
            }
        }
        attributes.emplace_back(AttributeDescription(point_data->GetArrayName(i), type), data);

        std::cout << "Attribute: " << attributes.back().desc.name
                  << ", # attribs: " << arr->GetNumberOfValues()
                  << ", data type: " << print_data_type(type) << "\n";
    }

    std::vector<glm::vec3> points;
    for (size_t i = 0; i < mesh->GetNumberOfPoints(); ++i) {
        double p[3] = {0};
        mesh->GetPoint(i, p);
        points.push_back(glm::vec3(p[0], p[1], p[2]));
    }

    BATree tree = LBATreeBuilder(std::move(points), std::move(attributes)).compact();
    write_ba_tree(args[2], tree);

    return 0;
}
