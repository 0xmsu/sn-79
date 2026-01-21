find_package(GTest CONFIG REQUIRED)

set(TEST_SOURCES "" CACHE INTERNAL "List of all collected test compilation units")

include(CollectTestSources)

add_subdirectory(test-common)
add_subdirectory(accounting)
add_subdirectory(decimal)
add_subdirectory(message)
add_subdirectory(serialization)
add_subdirectory(util)

add_executable(taosim-tests
    ${TEST_SOURCES}
    UtilTests.cpp
    SelfTradePreventionTests.cpp
    QuoteBasedOrderTests.cpp
    LoanSettlementTests.cpp
)

target_include_directories(taosim-tests
    PRIVATE
        "${src_dir}/net"
        "${src_dir}/util"
        test-common
)

target_link_libraries(taosim-tests
    PRIVATE
        GTest::gmock_main
        taosim_tests::common
        taosim::simulator
)

add_test(Tests taosim-tests)
