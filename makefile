NAME=bin/neuroscapes
SRC=neunet matrix util

ODIR=bin
SDIR=src

CXX=g++
CPPFLAGS=-g -MP -MD -fPIC -I $(SDIR)

LDFLAGS=-g -fPIC -I $(SDIR)
LDLIBS=

OBJS=$(SRC:%=$(ODIR)/%.o)
SRCS=$(SRC:%=$(SDIR)/%.cpp)
DEPS=$(SRC:%=$(ODIR)/%.d)

-include $(DEPS)
	
$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $<
	
$(NAME): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: shared
shared: $(OBJS)
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean: 
	rm -f $(OBJS) $(DEPS) $(NAME) $(NAME).so $(NAME).dll
