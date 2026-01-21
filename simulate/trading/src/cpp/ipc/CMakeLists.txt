file(GLOB ipc_sources CONFIGURE_DEPENDS "*.cpp")
add_library(ipc STATIC ${ipc_sources})
add_library(taosim::ipc ALIAS ipc)

target_include_directories(ipc PUBLIC include)

find_library(RT_LIBRARY rt REQUIRED)
target_link_libraries(ipc
    PRIVATE
        ${RT_LIBRARY}
        fmt::fmt
)

target_compile_options(ipc PUBLIC
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-missing-field-initializers>
)