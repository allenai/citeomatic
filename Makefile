ROOT=.

docker-citeomatic: docker-server-base sync-deps # client-js
	cd ${ROOT} && docker build\
		--tag citeomatic-server.base\
		-f ./citeomatic/docker/Dockerfile.base .

	cd ${ROOT} && docker build\
		--tag citeomatic-server.cpu\
		-f ./citeomatic/docker/Dockerfile.cpu .

	cd ${ROOT} && docker build\
		--tag citeomatic-server.gpu\
		-f ./citeomatic/docker/Dockerfile.gpu .


sync-deps: citeomatic/requirements.in base/requirements.in
	cd ${ROOT} && ./scripts/sync-project --project=base --op=compile
	cd ${ROOT} && ./scripts/sync-project --project=citeomatic --op=compile

client-js:
	cd ${ROOT}/citeomatic/client && npm install && npm run build

export-to-s3: docker-citeomatic
	./docker/push_to_ecr.sh

include base/Makefile
