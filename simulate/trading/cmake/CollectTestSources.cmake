function(collect_test_sources)
    file(GLOB local_test_sources CONFIGURE_DEPENDS "*.cpp")
    set(TEST_SOURCES
        "${TEST_SOURCES};${local_test_sources}"
        CACHE INTERNAL
        "List of all collected test compilation units")
endfunction()
