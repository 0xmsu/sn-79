file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")

add_executable(msgpack_tests ${sources})

target_include_directories(msgpack_tests PRIVATE include)

target_link_libraries(msgpack_tests
    PRIVATE
        GTest::gmock_main
        fmt::fmt
        taosim::serialization
)

add_test(MsgPackTests msgpack_tests)