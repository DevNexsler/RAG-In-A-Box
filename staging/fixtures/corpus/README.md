# Standing staging corpus

A small, committed set of representative documents for a **parallel,
testing-only** staging environment — never production data. Drop your own
representative PDFs, images, audio/video, and notes here; `scripts/seed_staging.sh`
deposits everything in this directory into a running staging stack's
`/data/documents` and triggers a per-file index.

This is **opt-in**: the hermetic gate e2e (`make test-e2e`) never touches this
corpus and always starts from an empty index. Seeding is only for manual /
cross-repo (CDS) testing against a standing dataset.

**Persistence:** the staging volumes are throwaway. To keep a seeded corpus
across restarts, stop the stack with `down` (NOT `down -v`) — `down -v` wipes
the volumes. Bring it back with `up` and the seeded docs (and their index) are
still there.

Nothing here is production data: the staging stack mounts its own throwaway
Docker volume, not the production documents directory.
