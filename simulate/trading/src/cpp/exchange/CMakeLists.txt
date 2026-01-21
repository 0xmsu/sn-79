file(GLOB exchange_sources CONFIGURE_DEPENDS "*.cpp")
add_library(exchange STATIC ${exchange_sources})
add_library(taosim::exchange ALIAS exchange)

target_include_directories(exchange PUBLIC include)

target_link_libraries(exchange
    PUBLIC
        taosim::accounting
        taosim::book
        taosim::decimal
        taosim::event
        taosim::logging
        taosim::message
        taosim::process
        taosim::replay
        taosim::serialization
        taosim::simulation
)
