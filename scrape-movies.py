#!/usr/bin/env/ python3

# scrape-movies.py - retreive movie database from tmdb
# movies are collected a year at a time
# API key can be retreived by signing up at
    # https://developers.themoviedb.org/3/getting-started/introduction


import pandas as pd
import tmdbsimple as tmdb
import numpy as np
import time

tmdb.API_KEY = 'Signup for a key at TMDB'
retry_time = 30
dbFileName = 'moviesDb-raw-test.csv'


def getCertificationUS(id):
    response = id.releases()
    for c in id.countries:
        if c['iso_3166_1'] == 'US':
            return c['certification']

def getMovie(discoverId):

    while(True):
        try:
            id = tmdb.Movies(discoverId)
            response = id.info()
            certification_US = getCertificationUS(id)
            break
        except Exception as e:
            print(e)
            print(f'Error when getting movie, retrying in {retry_time} secs')
            time.sleep(retry_time)

    # add movie to dataframe
    movieRow = pd.DataFrame([[discoverId,
                              id.title,
                              id.original_title,
                              id.release_date,
                              id.budget,
                              id.revenue,
                              id.popularity,
                              id.runtime,
                              id.vote_average,
                              id.vote_count,
                              id.adult,
                              id.status,
                              certification_US,
                              id.genres,
                              id.production_companies,
                              id.production_countries]])

    print(f'\t{id.release_date} {id.title}')
    return movieRow


def getMoviesByYear(year, dbFileName):

    # get total pages from discovery query
    while(True):
        try:
            discover = tmdb.Discover()
            discoverResponse = discover.movie(page = 1, language='en-US',
                sort_by='release_date.asc', include_adult=True, primary_release_year=year)
            numPages = discover.total_pages
            break
        except:
            print(f'\tError when getting page number, retrying in {retry_time} secs')
            time.sleep(retry_time)


    # iterate through each page and get info
    for currentPage in range(1, numPages+1):
        print(f'Page: {currentPage} / {numPages}')

        while(True):
            try:
                discover = tmdb.Discover()
                discoverResponse = discover.movie(page = currentPage, language='en-US',
                    sort_by='release_date.asc', include_adult=True, primary_release_year=year)
                break
            except:
                print(f'Error when getting page, retying in {retry_time} secs')
                time.sleep(retry_time)


        for item in discover.results:

            discoverId = item['id']

            # get Movies by id because discover does not include all info
            movieRow = getMovie(discoverId)
            # moviesDf = moviesDf.append(movieRow)

            # append movie to csv file
            movieRow.to_csv(dbFileName, mode='a', header=False, encoding='utf-8')



def main():

    # initialize DataFrame
    moviesDf = pd.DataFrame(columns=['id',
                                     'title',
                                     'original_title',
                                     'release_date',
                                     'budget',
                                     'revenue',
                                     'popularity',
                                     'runtime',
                                     'vote_average',
                                     'vote_count',
                                     'adult',
                                     'status',
                                     'certification_US',
                                     'genres',
                                     'production_companies',
                                     'production_countries'])
    moviesDf.to_csv(dbFileName, mode='a', encoding='utf-8')
    for year in range(1873, 2020):
        getMoviesByYear(year, dbFileName)



if __name__ == '__main__':
    main()
