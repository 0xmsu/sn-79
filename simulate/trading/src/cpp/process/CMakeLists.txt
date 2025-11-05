add_library(process STATIC
    GBM.cpp
    JumpDiffusion.cpp
    FundamentalPrice.cpp
    FuturesSignal.cpp
    Process.hpp
    ProcessFactory.cpp
    RNG.cpp
)
add_library(taosim::process ALIAS process)

target_include_directories(process
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${src_dir}"
        "${src_dir}/accounting"
        "${src_dir}/agent"
        "${src_dir}/base"
        "${src_dir}/book"
        "${src_dir}/decimal"
        "${src_dir}/exchange"
        "${src_dir}/mp"
        "${src_dir}/net"
        "${src_dir}/serialization"
        "${src_dir}/util"
)

target_link_libraries(process
    PRIVATE
        taosim::exchange
        taosim::message
        taosim::replay
        taosim::simulation
        pybind11::embed
)
