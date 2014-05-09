#Author : Witold Wolski

include_directories(${inc})
link_directories(${STATS_LIBRARY} ${VIGRA_LIBRARY} ${Glog_LIBRARY})

ADD_EXECUTABLE(atroustest atrousTest.cpp )
TARGET_LINK_LIBRARIES(atroustest  ${Boost_LIBRARIES} ${vigralibs} glog )


ENABLE_TESTING()
add_test(NAME atroustest
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
COMMAND ${BinOut}/atroustest )
set_tests_properties(atroustest PROPERTIES ENVIRONMENT "${LD_VARNAME}=${LD_PATH}")
