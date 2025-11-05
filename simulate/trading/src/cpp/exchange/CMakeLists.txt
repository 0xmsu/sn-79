add_library(exchange STATIC
    ClearingManager.cpp
    Exchange.cpp
    ExchangeConfig.cpp
    ExchangeSignals.cpp
    include/taosim/exchange/FeePolicy.hpp
    TieredFeePolicy.cpp
    FeePolicyWrapper.cpp
    DynamicFeePolicy.cpp
    OrderPlacementValidator.cpp
    ReplayEventLogger.cpp
)
add_library(taosim::exchange ALIAS exchange)

target_include_directories(exchange
    PUBLIC
        include
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${src_dir}"
        "${src_dir}/accounting"
        "${src_dir}/agent"
        "${src_dir}/book"
        "${src_dir}/base"
        "${src_dir}/decimal"
        "${src_dir}/exchange"
        "${src_dir}/mp"
        "${src_dir}/net"
        "${src_dir}/process"
        "${src_dir}/serialization"
        "${src_dir}/util"
)

target_link_libraries(exchange
    PUBLIC
        taosim::accounting
        taosim::book
        taosim::event
        taosim::message
        taosim::replay
        taosim::simulation
        bdl
        pybind11::embed
)
