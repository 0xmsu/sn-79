add_library(serialization STATIC
    CheckpointSerializable.hpp
    JsonSerializable.hpp
    json_util.cpp
)
add_library(taosim::serialization ALIAS serialization)

target_include_directories(serialization
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(serialization
    PUBLIC
        taosim::decimal
        taosim::mp
)
