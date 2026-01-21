add_library(common STATIC
    formatting.hpp
)
add_library(taosim_tests::common ALIAS common)

target_link_libraries(common
    PRIVATE
        taosim::decimal
)
