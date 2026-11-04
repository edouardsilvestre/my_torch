NAME_ANALYZER	=	my_torch_analyzer
SRC_ANALYZER	=	bin/my_torch_analyzer.py

all:	$(NAME_ANALYZER)

$(NAME_ANALYZER):	$(SRC_ANALYZER)
	cp $(SRC_ANALYZER) $(NAME_ANALYZER)
	chmod +x $(NAME_ANALYZER)

tests_run:	$(NAME_ANALYZER)
	python3 -m pytest tests/

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache

fclean:	clean
	rm -f $(NAME_ANALYZER)

re:	fclean all

.PHONY:	all clean fclean re tests_run
