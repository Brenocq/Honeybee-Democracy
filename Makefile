CC = g++
# Text style
RED    = \033[0;31m
GREEN  = \033[0;32m
NC     = \033[0m
BOLD   = \033[1m

# Folders
BIN	   = bin/
SRC	   = src/
LIB    = src/
OBJ    = obj/

# Files
FILES = scoutBee environment

SOURCES=$(patsubst %, ${SRC}%.cpp, ${FILES})
HEADERS=$(patsubst %, ${LIB}%.h, ${FILES})
OBJECTS=$(patsubst %, ${OBJ}%.o, ${FILES})

DEPENDENCIES=${LIB}parameters.h

EXECUTABLE=${BIN}beeSimulation

# Flags
FLAGS= -lGL -lGLU -lglut -Wall -I${LIB}

# Targets
${EXECUTABLE}: ${OBJECTS} ${DEPENDENCIES}
	@/bin/echo -e "${GREEN}${BOLD}----- Creating executable -----${NC}"
	${CC} -g ${SRC}main.cpp -o ${EXECUTABLE} ${FLAGS} ${OBJECTS} 

# Compile project files
${OBJ}%.o: ${SRC}%.cpp
	@/bin/echo -e "${GREEN}Compiling $<${NC}"
	${CC} -c $< -o $@ ${FLAGS} 

clean:
	@/bin/echo -e "${GREEN}${BOLD}----- Cleaning project -----${NC}"
	rm -rf ${OBJ}*.o
	rm -rf ${EXECUTABLE}

run: ${EXECUTABLE}
	@/bin/echo -e "${GREEN}${BOLD}----- Running ${EXECUTABLE} -----${NC}"
	./${EXECUTABLE}
