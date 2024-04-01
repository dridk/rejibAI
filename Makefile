


index: 
	rm -Rf chroma_db
	python index.py 

rag:
	chainlit run app.py -w -h
	
