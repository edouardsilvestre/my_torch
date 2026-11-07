NAME_ANALYZER	=	my_torch_analyzer
SRC_ANALYZER	=	bin/my_torch_analyzer.py

NAME_GENERATOR	=	my_torch_generator
SRC_GENERATOR	=	bin/my_torch_generator.py

all:	$(NAME_ANALYZER) $(NAME_GENERATOR)

$(NAME_ANALYZER):	$(SRC_ANALYZER)
	cp $(SRC_ANALYZER) $(NAME_ANALYZER)
	chmod +x $(NAME_ANALYZER)

$(NAME_GENERATOR):	$(SRC_GENERATOR)
	cp $(SRC_GENERATOR) $(NAME_GENERATOR)
	chmod +x $(NAME_GENERATOR)

tests_run:	$(NAME_ANALYZER) $(NAME_GENERATOR)
	python3 -m pytest tests/

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache

fclean:	clean
	rm -f $(NAME_ANALYZER) $(NAME_GENERATOR)

re:	fclean all

.PHONY:	all clean fclean re tests_run
