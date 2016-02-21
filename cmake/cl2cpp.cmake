if (NOT EXISTS "${CL_DIR}")
  message(FATAL_ERROR "Specified wrong OpenCL kernels directory: ${CL_DIR}")
endif()

file(GLOB cl_list "${CL_DIR}/*.cl" )
list(SORT cl_list)

if (NOT cl_list)
  message(FATAL_ERROR "Can't find OpenCL kernels in directory: ${CL_DIR}")
endif()

string(REPLACE ".cpp" ".hpp" OUTPUT_HPP "${OUTPUT}")
get_filename_component(OUTPUT_HPP_NAME "${OUTPUT_HPP}" NAME)

if("${MODULE_NAME}" STREQUAL "ocl")
    set(nested_namespace_start "")
    set(nested_namespace_end "")
else()
    set(new_mode ON)
    set(nested_namespace_start "namespace ${MODULE_NAME}\n{")
    set(nested_namespace_end "}")
endif()

set(STR_CPP "// This file is auto-generated. Do not edit!

#include \"precomp.hpp\"
#include \"cvconfig.h\"
#include \"${OUTPUT_HPP_NAME}\"

#ifdef HAVE_OPENCL

namespace cv
{
namespace ocl
{
${nested_namespace_start}

")

set(STR_HPP "// This file is auto-generated. Do not edit!

#include \"opencv2/core/ocl.hpp\"
#include \"opencv2/core/ocl_genbase.hpp\"
#include \"opencv2/core/opencl/ocl_defs.hpp\"

#ifdef HAVE_OPENCL

namespace cv
{
namespace ocl
{
${nested_namespace_start}

")

# set (Python_ADDITIONAL_VERSIONS 2.6 2.7)
find_package(PythonInterp)

if (NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "Python Interpreter Not Found.")
else()
    message("Python Interpreter Found. Version: ${PYTHON_VERSION_STRING}")
endif()

foreach(cl ${cl_list})
  get_filename_component(cl_filename "${cl}" NAME_WE)
  #message("${cl_filename}")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" "
import sys;
import base64;
SALT = '85W MagSage 2 Power Adapter';
xor_salt = lambda s: ''.join([chr(ord(c) ^ ord(SALT[i % len(SALT)])) for i,c in enumerate(s)]);
ret = base64.b64encode(xor_salt(sys.stdin.read()).encode('ascii')).decode('ascii');
oneline = lambda s: ', '.join([str(ord(x)) for x in s]);
print(',\\n'.join([oneline(ret[i:i+20]) for i in range(0, len(ret), 20)])+', 0');"
                  OUTPUT_VARIABLE lines
                  INPUT_FILE "${cl}")
  # file(READ "${cl}" lines)

  # string(REPLACE "\r" "" lines "${lines}\n")
  # string(REPLACE "\t" "  " lines "${lines}")

  # string(REGEX REPLACE "/\\*([^*]/|\\*[^/]|[^*/])*\\*/" ""   lines "${lines}") # multiline comments
  # string(REGEX REPLACE "/\\*([^\n])*\\*/"               ""   lines "${lines}") # single-line comments
  # string(REGEX REPLACE "[ ]*//[^\n]*\n"                 "\n" lines "${lines}") # single-line comments
  # string(REGEX REPLACE "\n[ ]*(\n[ ]*)*"                "\n" lines "${lines}") # empty lines & leading whitespace
  # string(REGEX REPLACE "^\n"                            ""   lines "${lines}") # leading new line

  # string(REPLACE "\\" "\\\\" lines "${lines}")
  # string(REPLACE "\"" "\\\"" lines "${lines}")
  # string(REPLACE "\n" "\\n\"\n\"" lines "${lines}")
  # string(REPLACE "\n" "\"\n\"" lines "${lines}")

  # string(REGEX REPLACE "\"$" "" lines "${lines}") # unneeded " at the eof

  string(MD5 hash "${lines}")

  set(STR_CPP_DECL "static const char ${cl_filename}_src[] = {\n${lines}\n};\n const struct ProgramEntry ${cl_filename}={\"${cl_filename}\",\n${cl_filename}_src, \"${hash}\"};\n")
  set(STR_HPP_DECL "extern const struct ProgramEntry ${cl_filename};\n")
  if(new_mode)
    set(STR_CPP_DECL "${STR_CPP_DECL}ProgramSource ${cl_filename}_oclsrc(${cl_filename}.programStr);\n")
    set(STR_HPP_DECL "${STR_HPP_DECL}extern ProgramSource ${cl_filename}_oclsrc;\n")
  endif()

  set(STR_CPP "${STR_CPP}${STR_CPP_DECL}")
  set(STR_HPP "${STR_HPP}${STR_HPP_DECL}")
endforeach()

set(STR_CPP "${STR_CPP}}\n${nested_namespace_end}}\n#endif\n")
set(STR_HPP "${STR_HPP}}\n${nested_namespace_end}}\n#endif\n")

file(WRITE "${OUTPUT}" "${STR_CPP}")

if(EXISTS "${OUTPUT_HPP}")
  file(READ "${OUTPUT_HPP}" hpp_lines)
endif()
if("${hpp_lines}" STREQUAL "${STR_HPP}")
  message(STATUS "${OUTPUT_HPP} contains same content")
else()
  file(WRITE "${OUTPUT_HPP}" "${STR_HPP}")
endif()
