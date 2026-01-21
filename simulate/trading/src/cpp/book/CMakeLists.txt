file(GLOB book_sources CONFIGURE_DEPENDS "*.cpp")
add_library(book STATIC ${book_sources})
add_library(taosim::book ALIAS book)

target_include_directories(book
    PUBLIC
        include
        "${SRC_DIR}/agent"
        "${SRC_DIR}/base"
        "${SRC_DIR}/mp"
        "${SRC_DIR}/net"
        "${SRC_DIR}/serialization"
        "${SRC_DIR}/util"
)

target_link_libraries(book
    PUBLIC
        taosim::exchange
        taosim::logging
        taosim::message
        taosim::process
        fmt::fmt
)
