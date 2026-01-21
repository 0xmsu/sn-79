add_library(base STATIC
    IConfigurable.hpp
    CSVPrintable.hpp
    IHumanPrintable.hpp
    IMessageable.cpp
    IPrintable.hpp
)
add_library(taosim::base ALIAS base)

target_include_directories(base
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SRC_DIR}"
        "${SRC_DIR}/accounting"
        "${SRC_DIR}/agent"
        "${SRC_DIR}/book"
        "${SRC_DIR}/decimal"
        "${SRC_DIR}/exchange"
        "${SRC_DIR}/message"
        "${SRC_DIR}/mp"
        "${SRC_DIR}/net"
        "${SRC_DIR}/serialization"
        "${SRC_DIR}/util"
)

target_link_libraries(base
    PRIVATE
        taosim::exchange
        taosim::process
        pugixml::static pugixml::pugixml
        pybind11::embed
    INTERFACE
        taosim::mp
)
