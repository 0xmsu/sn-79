file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
add_library(simulation STATIC ${sources})
add_library(taosim::simulation ALIAS simulation)

target_include_directories(simulation
    PUBLIC
        include
        "${src_dir}"
        "${src_dir}/accounting"
        "${src_dir}/accounting/include"
        "${src_dir}/agent"
        "${src_dir}/base"
        "${src_dir}/book"
        "${src_dir}/book/include"
        "${src_dir}/decimal"
        "${src_dir}/exchange"
        "${src_dir}/exchange/include"
        "${src_dir}/mp"
        "${src_dir}/net"
        "${src_dir}/process"
        "${src_dir}/util"
)

target_link_libraries(simulation
    PRIVATE
        taosim::decimal
        taosim::event
        taosim::ipc
        taosim::message
        taosim::net
        taosim::replay
        taosim::serialization
        Boost::boost
        pybind11::embed
)