UNLOAD (
    SELECT
        id,
        MAX(title) AS title,
        MAX(abstract) AS abstract,
        MAX(year) AS year,
        NORMALIZE(
            ARRAY_JOIN(
                ARRAY_AGG(paragraph), chr(13) || chr(13)
            )
        ) AS full_text
    FROM (
        SELECT
            id,
            year,
            title,
            abstract,
            fields_of_study,
            SUBSTR(full_text, paralocs[1], paralocs[2] - paralocs[1] + 1) as paragraph
        FROM (
            SELECT
                id,
                year,
                title,
                abstract,
                full_text,
                ARRAY_JOIN(fields_of_study, ',') as fields_of_study,
                ARRAY_SORT(
                    ARRAY_DISTINCT(
                        TRANSFORM(
                            all_paralocs,
                            x -> (
                                ARRAY [CAST(JSON_EXTRACT(x, '$.start') AS INT)] ||
                                ARRAY [CAST(JSON_EXTRACT(x, '$.end') AS INT)]
                            )
                        )
                    )
                ) AS all_paralocs
            FROM (
                SELECT
                    spr.id,
                    spr.metadata.publication_date.year as year,
                    spr.metadata.title as title,
                    spr.metadata.abstract as abstract,
                    CAST (
                        json_parse(spr.metadata.fields_of_study) AS array(varchar)
                    ) AS fields_of_study,
                    spr.content.grobid.contents as full_text,
                    CAST(
                        json_parse(spr.content.grobid.annotations.paragraph)
                        AS array(json)
                    ) || CAST(
                        json_parse(spr.content.grobid.annotations.section_header)
                        AS array(json)
                    ) AS all_paralocs
                FROM (
                    SELECT corpus_paper_id as id
                    FROM content_ext.paper_sources
                    WHERE source = 'ACL'
                ) AS corpus_ext
                INNER JOIN s2orc_papers.releases AS spr
                    ON spr.id = corpus_ext.id
                WHERE
                    year=2022 AND
                    month=12 AND
                    day=11 AND
                    metadata.fields_of_study IS NOT NULL
            )
            WHERE CONTAINS(fields_of_study, 'Computer Science')
        )
        CROSS JOIN UNNEST(all_paralocs) as t(paralocs)
    )
    GROUP BY id
)
TO 's3://ai2-s2-lucas/s2orc_20221211/acl_s2orc/'
WITH (
    format='JSON',
    compression='GZIP'
)
