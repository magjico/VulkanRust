@echo off
set GLSLC=C:\VulkanSDK\1.3.268.0\Bin\glslc.exe
set SHADER_PATH=vulkan-tutorial\shaders\hello_triangle

"%GLSLC%" "%SHADER_PATH%"\shader.vert -o "%SHADER_PATH%"\vert.spv
"%GLSLC%" "%SHADER_PATH%"\shader.frag -o "%SHADER_PATH%"\frag.spv
pause